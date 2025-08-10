#!/usr/bin/env python3
"""
Build a large conversation corpus from popular open-source datasets and
write to datasets/large_conversation_training.txt in "Human:/Assistant:" format.

Datasets used (review and comply with individual licenses):
  - teknium/OpenHermes-2.5 (or brahmairesearch/OpenHermes-2.5-Formatted fallback)
  - HuggingFaceH4/ultrachat_200k
  - Open-Orca/OpenOrca (or Open-Orca/SlimOrca fallback)

Usage:
  python scripts/build_conversation_corpus.py [--max_per_source N]

Requires:
  pip install -r requirements.txt  # includes datasets, tokenizers
"""

import argparse
import os
from typing import Iterable, List, Tuple

from datasets import load_dataset


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_pairs(f, pairs: Iterable[Tuple[str, str]]) -> int:
    """Write (user, assistant) pairs in the required format. Returns count written."""
    written = 0
    for user_text, assistant_text in pairs:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if user_text and assistant_text:
            f.write(f"Human: {user_text}\nAssistant: {assistant_text}\n\n")
            written += 1
    return written


def extract_pairs_from_messages(messages) -> List[Tuple[str, str]]:
    """Convert a chat-style list of dicts into (user, assistant) pairs."""
    pairs: List[Tuple[str, str]] = []
    pending_user: str | None = None
    for msg in messages or []:
        role = (msg.get("role") or msg.get("from") or "").lower()
        text = msg.get("content") or msg.get("value") or msg.get("text") or ""
        if role in ("user", "human"):
            pending_user = text
        elif role == "assistant" and pending_user:
            pairs.append((pending_user, text))
            pending_user = None
    return pairs


def add_openhermes(f, limit: int | None = None) -> int:
    """Add pairs from OpenHermes 2.5 (with a formatted fallback). Returns blocks written."""
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    except Exception:
        ds = load_dataset("brahmairesearch/OpenHermes-2.5-Formatted", split="train")

    blocks = 0
    for row in ds:
        conversations = row.get("conversations") or row.get("messages") or []
        pairs = extract_pairs_from_messages(conversations)
        if pairs:
            if write_pairs(f, pairs) > 0:
                blocks += 1
                if limit and blocks >= limit:
                    break
    return blocks


def add_ultrachat(f, limit: int | None = None) -> int:
    """Add pairs from UltraChat-200k. Returns blocks written."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    blocks = 0
    for row in ds:
        conversations = row.get("messages") or row.get("conversations") or []
        pairs = extract_pairs_from_messages(conversations)
        if pairs:
            if write_pairs(f, pairs) > 0:
                blocks += 1
                if limit and blocks >= limit:
                    break
    return blocks


def add_openorca(f, limit: int | None = None) -> int:
    """Add pairs from OpenOrca (fallback to SlimOrca). Returns blocks written."""
    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train")
    except Exception:
        ds = load_dataset("Open-Orca/SlimOrca", split="train")

    blocks = 0
    for row in ds:
        system_prompt = row.get("system_prompt") or ""
        question = row.get("question") or row.get("user") or ""
        response = row.get("response") or row.get("assistant") or ""
        user_text = (system_prompt + "\n" + question).strip() if system_prompt else question
        if user_text and response:
            if write_pairs(f, [(user_text, response)]) > 0:
                blocks += 1
                if limit and blocks >= limit:
                    break
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conversation corpus from open datasets")
    parser.add_argument(
        "--max_per_source",
        type=int,
        default=None,
        help="Limit number of conversation blocks per dataset (for quick runs)",
    )
    args = parser.parse_args()

    ensure_dir("datasets")
    out_path = os.path.join("datasets", "large_conversation_training.txt")

    total_blocks = 0
    with open(out_path, "w", encoding="utf-8") as f:
        print("Adding OpenHermes 2.5 …")
        total_blocks += add_openhermes(f, args.max_per_source)

        print("Adding UltraChat-200k …")
        total_blocks += add_ultrachat(f, args.max_per_source)

        print("Adding OpenOrca …")
        total_blocks += add_openorca(f, args.max_per_source)

    print(f"\nWrote conversation corpus to {out_path}")
    print(f"Blocks written (conversations): {total_blocks}")
    print("Done.")


if __name__ == "__main__":
    main()


