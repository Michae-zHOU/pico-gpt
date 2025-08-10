#!/usr/bin/env python3
"""
Analyze and categorize datasets to identify duplicates and usefulness
"""

import os
import hashlib
from collections import defaultdict

def get_file_info(filepath):
    """Get file information including size and content hash"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        size = len(content)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Analyze content type
        lines = content.split('\n')
        human_lines = sum(1 for line in lines if 'Human:' in line)
        assistant_lines = sum(1 for line in lines if 'Assistant:' in line)
        
        # Sample first few lines
        sample = '\n'.join(lines[:5])
        
        return {
            'file': os.path.basename(filepath),
            'size': size,
            'hash': content_hash,
            'human_lines': human_lines,
            'assistant_lines': assistant_lines,
            'total_lines': len(lines),
            'sample': sample
        }
    except Exception as e:
        return {
            'file': os.path.basename(filepath),
            'error': str(e)
        }

def main():
    datasets_dir = 'datasets'
    if not os.path.exists(datasets_dir):
        print("Datasets directory not found")
        return
    
    # Analyze all datasets
    dataset_info = []
    txt_files = [f for f in os.listdir(datasets_dir) if f.endswith('.txt')]
    
    print("DATASET ANALYSIS")
    print("=" * 50)
    
    for filename in txt_files:
        filepath = os.path.join(datasets_dir, filename)
        info = get_file_info(filepath)
        dataset_info.append(info)
    
    # Sort by size
    dataset_info.sort(key=lambda x: x.get('size', 0), reverse=True)
    
    # Display analysis
    print(f"{'File':<35} {'Size':<10} {'Hash':<10} {'Conversations':<15} {'Type'}")
    print("-" * 85)
    
    hash_groups = defaultdict(list)
    
    for info in dataset_info:
        if 'error' in info:
            print(f"{info['file']:<35} ERROR: {info['error']}")
            continue
            
        size_mb = info['size'] / 1024 if info['size'] > 1024 else info['size']
        size_unit = 'KB' if info['size'] > 1024 else 'B'
        
        conversations = min(info['human_lines'], info['assistant_lines'])
        
        # Categorize dataset type
        if 'reasoning' in info['file'].lower():
            dataset_type = 'Reasoning'
        elif 'large' in info['file'].lower():
            dataset_type = 'Large Conv'
        elif 'literature' in info['file'].lower():
            dataset_type = 'Literature'
        elif 'conversation' in info['file'].lower():
            dataset_type = 'Conversation'
        elif 'smart' in info['file'].lower():
            dataset_type = 'Smart'
        else:
            dataset_type = 'Other'
        
        print(f"{info['file']:<35} {size_mb:>6.1f}{size_unit:<3} {info['hash']:<10} {conversations:<15} {dataset_type}")
        
        # Group by hash to find duplicates
        hash_groups[info['hash']].append(info)
    
    # Find duplicates
    print(f"\nDUPLICATE DETECTION")
    print("=" * 30)
    
    duplicates_found = False
    for content_hash, files in hash_groups.items():
        if len(files) > 1:
            duplicates_found = True
            print(f"Duplicate content (hash {content_hash}):")
            for file_info in files:
                print(f"  - {file_info['file']} ({file_info['size']} bytes)")
            print()
    
    if not duplicates_found:
        print("No exact duplicates found based on content hash.")
    
    # Categorize by purpose
    print(f"\nDATASET CATEGORIES")
    print("=" * 30)
    
    categories = {
        'conversation': [],
        'reasoning': [],
        'literature': [],
        'large_datasets': [],
        'small_test': []
    }
    
    for info in dataset_info:
        if 'error' in info:
            continue
            
        filename = info['file'].lower()
        size = info['size']
        
        if 'reasoning' in filename:
            categories['reasoning'].append(info)
        elif 'literature' in filename:
            categories['literature'].append(info)
        elif size > 100000:  # > 100KB
            categories['large_datasets'].append(info)
        elif size < 10000:   # < 10KB
            categories['small_test'].append(info)
        else:
            categories['conversation'].append(info)
    
    for category, files in categories.items():
        if files:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for info in files:
                print(f"  - {info['file']} ({info['size']} bytes)")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS")
    print("=" * 30)
    
    # Find the best datasets to keep
    keep_files = []
    
    # Keep the largest conversation dataset
    large_conv = [info for info in dataset_info if info.get('size', 0) > 500000 and 'conversation' in info['file'].lower()]
    if large_conv:
        best_large = max(large_conv, key=lambda x: x['size'])
        keep_files.append(best_large['file'])
        print(f"KEEP: {best_large['file']} - Primary large conversation dataset ({best_large['size']} bytes)")
    
    # Keep reasoning dataset
    reasoning_files = [info for info in dataset_info if 'reasoning' in info['file'].lower()]
    if reasoning_files:
        best_reasoning = max(reasoning_files, key=lambda x: x['size'])
        keep_files.append(best_reasoning['file'])
        print(f"KEEP: {best_reasoning['file']} - Reasoning training data ({best_reasoning['size']} bytes)")
    
    # Keep one clean conversation dataset
    clean_files = [info for info in dataset_info if 'clean' in info['file'].lower() or 'simple' in info['file'].lower()]
    if clean_files:
        best_clean = max(clean_files, key=lambda x: x['size'])
        if best_clean['file'] not in keep_files:
            keep_files.append(best_clean['file'])
            print(f"KEEP: {best_clean['file']} - Clean conversation data ({best_clean['size']} bytes)")
    
    # Keep literature if it's large and unique
    lit_files = [info for info in dataset_info if 'literature' in info['file'].lower()]
    if lit_files:
        best_lit = max(lit_files, key=lambda x: x['size'])
        if best_lit['size'] > 1000000:  # Only if > 1MB
            keep_files.append(best_lit['file'])
            print(f"KEEP: {best_lit['file']} - Literature dataset ({best_lit['size']} bytes)")
    
    # Files to remove
    all_files = [info['file'] for info in dataset_info if 'error' not in info]
    remove_files = [f for f in all_files if f not in keep_files]
    
    print(f"\nREMOVE CANDIDATES:")
    for filename in remove_files:
        file_info = next(info for info in dataset_info if info['file'] == filename)
        reason = "duplicate/redundant"
        if file_info['size'] < 1000:
            reason = "too small"
        elif 'backup' in filename:
            reason = "backup file"
        print(f"  - {filename} ({reason})")
    
    print(f"\nSUMMARY:")
    print(f"Total files: {len(all_files)}")
    print(f"Keep: {len(keep_files)} files")
    print(f"Remove: {len(remove_files)} files")
    print(f"Space saved: ~{sum(next(info for info in dataset_info if info['file'] == f)['size'] for f in remove_files)} bytes")

if __name__ == "__main__":
    main()