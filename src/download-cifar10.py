import sys
from utils import download_cifar10, load_cifar10_with_stats, verify_stats

if __name__ == "__main__":
    data_path = "../data"
    
    result = download_cifar10(data_path, compute_stats=True)
    if result['success']:
        print(f"\nComputed statistics:")
        print(f"Mean: {result['mean']}")
        print(f"Std: {result['std']}")
    
    train_loader, test_loader, stats = load_cifar10_with_stats(data_path)
    print(f"\nData loaded with stats:")
    print(f"Mean: {stats['mean']}")
    print(f"Std: {stats['std']}")
    
    verify_stats(data_path)