import argparse
import json
import numpy as np
from scipy.stats import ranksums
import os
import glob
from pathlib import Path

def get_latest_json(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.json"))
        if not files:
            raise ValueError(f"No JSON files found in {path}")
        return max(files, key=os.path.getmtime)
    raise ValueError(f"Path not found: {path}")

def load_data(path, method="stch_set"):
    json_path = get_latest_json(path)
    with open(json_path) as f:
        data = json.load(f)
    
    # Try to find the method
    methods = data.get("methods", {})
    # Fuzzy match if exact not found (e.g. stch_set_K5)
    if method not in methods:
        # try to find one that starts with method
        matches = [k for k in methods.keys() if k.startswith(method)]
        if matches:
            # Pick the first one or the one with "stch_set" exactly
            # If explicit match exists, use it.
            # If not, use the first match but warn.
            target = matches[0]
            # print(f"Warning: Method '{method}' not found in {json_path}, using '{target}'")
            method = target
        else:
            # Fallback: use the first method available
            target = list(methods.keys())[0]
            # print(f"Warning: Method '{method}' not found in {json_path}, using '{target}'")
            method = target

    hv_all = np.array(methods[method]["hv_all"])
    final_hv = hv_all[:, -1]
    return final_hv, method, json_path

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of BO results")
    parser.add_argument("path1", help="First result file or directory")
    parser.add_argument("path2", help="Second result file or directory")
    parser.add_argument("--method", default="stch_set", help="Method to compare (default: stch_set)")
    args = parser.parse_args()

    try:
        data1, name1, file1 = load_data(args.path1, args.method)
        data2, name2, file2 = load_data(args.path2, args.method)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"{'='*60}")
    print(f"Statistical Comparison")
    print(f"{'='*60}")
    print(f"Group 1: {name1} (from {os.path.basename(file1)})")
    print(f"Group 2: {name2} (from {os.path.basename(file2)})")
    print(f"-"*60)
    
    # Stats
    mean1, std1 = np.mean(data1), np.std(data1, ddof=1)
    mean2, std2 = np.mean(data2), np.std(data2, ddof=1)
    median1 = np.median(data1)
    median2 = np.median(data2)
    
    # Test
    # Wilcoxon rank-sum (Mann-Whitney U)
    stat, pval = ranksums(data1, data2)
    d = cohens_d(data1, data2)
    
    print(f"{'Metric':<15} {'Group 1':<15} {'Group 2':<15}")
    print(f"-"*60)
    print(f"{'Mean':<15} {mean1:<15.4f} {mean2:<15.4f}")
    print(f"{'Std':<15} {std1:<15.4f} {std2:<15.4f}")
    print(f"{'Median':<15} {median1:<15.4f} {median2:<15.4f}")
    print(f"{'N':<15} {len(data1):<15} {len(data2):<15}")
    print(f"-"*60)
    print(f"Wilcoxon Rank-Sum Test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value:   {pval:.4e}")
    print(f"  Cohen's d: {d:.4f}")
    
    if pval < 0.05:
        print(f"\nResult: SIGNIFICANT difference (p < 0.05)")
    else:
        print(f"\nResult: NO significant difference (p >= 0.05)")

if __name__ == "__main__":
    main()
