import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by swapping even/odd pairs for RoPE."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x.
    x: (B, n_head, T, head_dim); cos/sin: (T, head_dim) or broadcastable
    """
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_sdpa = getattr(config, 'use_sdpa', True)
        self.use_rope = getattr(config, 'use_rope', False)
        self.rope_base = getattr(config, 'rope_base', 10000.0)
        # Legacy bias mask for fallback path
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
        )

    def _get_rope_cos_sin(self, seq_start: int, seq_len: int, device: torch.device):
        t = torch.arange(seq_start, seq_start + seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        freqs = torch.einsum('t,d->td', t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        return cos, sin

    def forward(self, x, layer_past=None, use_cache: bool = False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        past_k, past_v = (None, None)
        if layer_past is not None:
            past_k, past_v = layer_past
            past_len = past_k.size(-2)

        if self.use_rope:
            cos, sin = self._get_rope_cos_sin(past_len, T, device=x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        if self.use_sdpa and hasattr(F, 'scaled_dot_product_attention') and past_len == 0:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            S = att.size(-1)
            if S == T:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            else:
                i = torch.arange(T, device=x.device).unsqueeze(-1)
                j = torch.arange(S, device=x.device).unsqueeze(0)
                mask = (i + past_len) >= j
                att = att.masked_fill(~mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        present = (k, v) if use_cache else None
        return y, present


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_mult = getattr(config, 'mlp_multiplier', 4)
        mlp_type = getattr(config, 'mlp_type', 'gelu')
        hidden = mlp_mult * config.n_embd
        self.mlp_type = mlp_type
        if mlp_type == 'swiglu':
            self.up_proj = nn.Linear(config.n_embd, hidden * 2, bias=config.bias)
            self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
            self.act = nn.SiLU()
        else:
            self.c_fc    = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.act     = nn.GELU()
            self.c_proj  = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.mlp_type == 'swiglu':
            up = self.up_proj(x)
            u, v = up.chunk(2, dim=-1)
            x = self.down_proj(self.act(u) * v)
        else:
            x = self.c_fc(x)
            x = self.act(x)
            x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None, use_cache: bool = False):
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # Modern options
    use_sdpa: bool = True
    use_rope: bool = False
    rope_base: float = 10000.0
    mlp_multiplier: int = 4
    mlp_type: str = 'gelu'


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kv=None, use_cache: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        tok_emb = self.transformer.wte(idx)
        if getattr(self.config, 'use_rope', False):
            x = tok_emb
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        x = self.transformer.drop(x)

        if past_kv is None:
            past_kv = [None] * len(self.transformer.h)
        presents = [] if use_cache else None

        for block, layer_past in zip(self.transformer.h, past_kv):
            x, present = block(x, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if use_cache:
            return logits, loss, presents
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=None):
        self.eval()
        past_kv = None
        device = idx.device
        input_ids = idx[:, -self.config.block_size:]

        out = self(input_ids, past_kv=None, use_cache=True)
        logits = out[0]
        past_kv = out[2]

        for _ in range(max_new_tokens):
            logits_step = logits[:, -1, :] / max(temperature, 1e-6)

            if repetition_penalty is not None and repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    token_ids = idx[b].unique()
                    token_logits = logits_step[b, token_ids]
                    positive = token_logits > 0
                    logits_step[b, token_ids[positive]] /= repetition_penalty
                    logits_step[b, token_ids[~positive]] *= repetition_penalty

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('inf')

            if top_p is not None and 0 < top_p < 1.0:
                probs = F.softmax(logits_step, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > top_p
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
            else:
                probs = F.softmax(logits_step, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            out = self(idx_next, past_kv=past_kv, use_cache=True)
            logits = out[0]
            past_kv = out[2]

        return idx