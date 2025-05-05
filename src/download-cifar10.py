import sys
from utils import download_cifar10, load_cifar10_with_stats, verify_stats

if __name__ == "__main__":
    data_path = "../data"
    
    # Method 1: Download and compute stats separately
    result = download_cifar10(data_path, compute_stats=True)
    if result['success']:
        print(f"\nComputed statistics:")
        print(f"Mean: {result['mean']}")
        print(f"Std: {result['std']}")
    
    # Method 2: Load data with automatic stats computation
    train_loader, test_loader, stats = load_cifar10_with_stats(data_path)
    print(f"\nData loaded with stats:")
    print(f"Mean: {stats['mean']}")
    print(f"Std: {stats['std']}")
    
    # Verify against standard values
    verify_stats(data_path)