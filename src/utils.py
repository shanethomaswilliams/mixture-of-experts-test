import torch # type: ignore
import torch.nn.functional as F # type: ignore
import tqdm # type: ignore
from sklearn.datasets import make_moons # type: ignore
from sklearn import metrics # type: ignore
import torch # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import numpy as np # type: ignore
from torchvision import datasets, transforms
import os
import json
import numpy as np
from tqdm import tqdm
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path
import sys

def train_model(model, device, tr_loader, va_loader, loss_module, batch_size=1,
                n_epochs=10, lr=0.001, l2pen_mag=0.0, data_order_seed=42,
                model_filename='best_model.pth',
                do_early_stopping=True,
                n_epochs_without_va_improve_before_early_stop=15,
                switch_balance_weight=0.1, importance_weight=0.05,
                topk_balance_weight=0.1, mutual_info_weight=0.05,
                entropy_weight=0.01):
    ''' Train model via stochastic gradient descent.

    Assumes provided model's trainable params already set to initial values.

    Returns
    -------
    best_model : PyTorch model
        Model corresponding to epoch with best validation loss (xent)
        seen at any epoch throughout this training run
    info : dict
        Contains history of this training run, for diagnostics/plotting
    '''
    model_name = model.__class__.__name__
    is_moe_model = ('MoE' in model_name or 'MOE' in model_name or 'Moe' in model_name)
    if is_moe_model:
        print(f"Detected MoE model: {model_name}. Using specialized MoE training.")
    else:
        print(f"Detected standard model: {model_name}. Using standard training.")
    
    # Make sure tr_loader shuffling reproducible
    torch.manual_seed(data_order_seed)      
    torch.cuda.manual_seed(data_order_seed)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr)

    # Allocate lists for tracking progress each epoch
    tr_info = {'xent':[], 'err':[], 'loss':[]}
    va_info = {'xent':[], 'err':[]}
    epochs = []
    
    # For tracking expert usage
    expert_usage_history = []

    # Init vars needed for early stopping
    best_va_loss = float('inf')
    curr_wait = 0 # track epochs we are waiting to early stop

    # Count size of datasets, for adjusting metric values to be per-example
    n_train = float(len(tr_loader.dataset))
    n_batch_tr = float(len(tr_loader))
    n_valid = float(len(va_loader.dataset))

    # Progress bar
    progressbar = tqdm(range(n_epochs + 1))
    pbar_info = {}

    # Curriculum learning for switch weight
    initial_switch_weight = switch_balance_weight  # Use provided weight as initial
    final_switch_weight = switch_balance_weight * 2.0  # Double it by the end
    switch_weight_schedule = np.linspace(initial_switch_weight, final_switch_weight, n_epochs)

    # Loop over epochs
    for epoch in progressbar:
        if epoch > 0:
            model.train() # In TRAIN mode
            tr_loss = 0.0  # aggregate total loss
            tr_xent = 0.0  # aggregate pure cross-entropy (without L2)
            tr_err = 0     # count mistakes on train set
            pbar_info['batch_done'] = 0
            
            # Get current switch weight from schedule
            current_switch_weight = switch_weight_schedule[epoch-1] if epoch <= n_epochs else final_switch_weight
            
            for bb, (x, y) in enumerate(tr_loader):
                optimizer.zero_grad()
                x_BF = x.to(device)
                y_B = y.to(device)

                logits_BC = model(x_BF)
                
                # Calculate pure cross-entropy for tracking
                pure_xent = loss_module.calc_xent_loss(logits_BC, y_B, reduction='mean')
                
                # Calculate cross-entropy with L2 for optimization
                xent_loss_with_l2 = loss_module.calc_xent_loss_with_l2(logits_BC, y_B, model, l2pen_mag, batch_size, reduction='mean')
                
                loss = xent_loss_with_l2
                
                if is_moe_model:
                    # Apply all load balancing losses
                    switch_lb_loss = loss_module.calc_switch_load_balancing_loss(model)
                    loss += current_switch_weight * switch_lb_loss

                    importance_loss = loss_module.calc_importance_loss_with_entropy(model)
                    loss += importance_weight * importance_loss
                    
                    if hasattr(model, 'top_k') and model.top_k > 1:
                        topk_lb_loss = loss_module.calc_topk_load_balancing_loss(model, k=model.top_k)
                        loss += topk_balance_weight * topk_lb_loss
                    
                    mi_loss = loss_module.calc_mutual_information_loss(model, y_B)
                    loss += mutual_info_weight * mi_loss
                    
                    # Track expert usage
                    if hasattr(model, 'last_gate_logits'):
                        _, expert_indices = torch.max(model.last_gate_logits, dim=1)
                        expert_counts = torch.bincount(expert_indices, minlength=model.num_experts)
                        expert_usage_history.append(expert_counts.cpu().numpy())

                loss.backward()
                optimizer.step()
    
                pbar_info['batch_done'] += 1        
                progressbar.set_postfix(pbar_info)
    
                # Increment loss metrics we track for debugging/diagnostics
                tr_loss += loss.item() / n_batch_tr
                tr_xent += pure_xent.item() / n_batch_tr  # Track pure cross-entropy only
                tr_err += metrics.zero_one_loss(
                    logits_BC.argmax(axis=1).detach().cpu().numpy(),
                    y_B.detach().cpu().numpy(), normalize=False)
                    
            tr_err_rate = tr_err / n_train
            
            # Print expert usage stats at end of epoch
            if is_moe_model and expert_usage_history:
                epoch_usage = np.mean(expert_usage_history, axis=0)
                print(f"Epoch {epoch} expert usage: {epoch_usage}")
                expert_usage_history = []  # Reset for next epoch
                
        else:
            # First epoch (0) doesn't train, just measures initial perf on val
            tr_loss = np.nan
            tr_xent = np.nan
            tr_err_rate = np.nan

        # Track performance on val set
        with torch.no_grad():
            model.eval() # In EVAL mode
            va_xent = 0.0
            va_err = 0
            for xva_BF, yva_B in va_loader:
                logits_BC = model(xva_BF.to(device))
                # For validation, we only track pure cross-entropy (no L2)
                va_xent += loss_module.calc_xent_loss(logits_BC, yva_B.to(device), reduction='sum').item()
                va_err += metrics.zero_one_loss(
                    logits_BC.argmax(axis=1).detach().cpu().numpy(),
                    yva_B, normalize=False)
            va_xent = va_xent / n_valid
            va_err_rate = va_err / n_valid

        # Update diagnostics and progress bar
        epochs.append(epoch)
        tr_info['loss'].append(tr_loss)
        tr_info['xent'].append(tr_xent)
        tr_info['err'].append(tr_err_rate)        
        va_info['xent'].append(va_xent)
        va_info['err'].append(va_err_rate)
        pbar_info.update({
            "tr_loss": tr_loss, "tr_xent": tr_xent, "tr_err": tr_err_rate,
            "va_xent": va_xent, "va_err": va_err_rate,
            })
        progressbar.set_postfix(pbar_info)

        # Early stopping logic
        # If loss is dropping, track latest weights as best
        if va_xent < best_va_loss:
            best_epoch = epoch
            best_va_loss = va_xent
            best_tr_err_rate = tr_err_rate
            best_va_err_rate = va_err_rate
            curr_wait = 0
            model = model.cpu()
            torch.save(model.state_dict(), model_filename)
            model.to(device)
        else:
            curr_wait += 1
                
        wait_enough = curr_wait >= n_epochs_without_va_improve_before_early_stop
        if do_early_stopping and wait_enough:
            print("Stopped early.")
            break

    print(f"Finished after epoch {epoch}, best epoch={best_epoch}")
    model.to(device)
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))    
    result = { 
        'data_order_seed':data_order_seed,
        'lr':lr, 'n_epochs':n_epochs, 'l2pen_mag':l2pen_mag,
        'tr':tr_info,
        'va':va_info,
        'best_tr_err': best_tr_err_rate,
        'best_va_err': best_va_err_rate,
        'best_va_loss': best_va_loss,
        'best_epoch': best_epoch,
        'epochs': len(epochs)}
    return model, result

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect outputs, labels, and predictions
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predicted.cpu())
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    accuracy = correct / total
    
    return accuracy, all_outputs, all_labels, all_predictions

def flatten_params(model, excluded_params=['lengthscale_param', 'outputscale_param', 'sigma_param']):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if name not in excluded_params])

def get_data_from_loader(loader):
    xs, ys = [], []
    for x_batch, y_batch in loader:
        xs.append(x_batch)
        ys.append(y_batch)
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    return x_all.numpy(), y_all.numpy()

def compute_l2_regularization(model, weight_decay=0.0001):
    """Compute L2 regularization term for all model parameters"""
    l2_reg = torch.tensor(0., device=next(model.parameters()).device)
    
    for name, param in model.named_parameters():
        if 'bias' not in name:  # Usually we don't regularize bias terms
            l2_reg += torch.norm(param, p=2)
    
    return weight_decay * l2_reg

def train_step_with_moe(model, images, labels, optimizer, lambda_balance=0.01, weight_decay=0.0001):
    """Training step for traditional MoE with cross entropy, L2 regularization, and load balancing"""
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(images)
    
    # Classification loss (Cross Entropy)
    ce_loss = F.cross_entropy(outputs, labels)
    
    # L2 Regularization
    l2_loss = compute_l2_regularization(model, weight_decay)
    
    # Load balancing loss
    balance_loss = model.get_load_balancing_loss()
    
    # Total loss = Cross Entropy + L2 Regularization + Load Balancing
    total_loss = ce_loss + l2_loss + lambda_balance * balance_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), ce_loss.item(), l2_loss.item(), balance_loss.item()

def train_step_with_moe_builtin_l2(model, images, labels, optimizer, lambda_balance=0.01):
    """Training step using optimizer's weight_decay for L2 regularization"""
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(images)
    
    # Classification loss (Cross Entropy)
    ce_loss = F.cross_entropy(outputs, labels)
    
    # Load balancing loss
    balance_loss = model.get_load_balancing_loss()
    
    # Total loss (L2 is handled by optimizer's weight_decay)
    total_loss = ce_loss + lambda_balance * balance_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), ce_loss.item(), balance_loss.item()

def calculate_dataset_stats(data_path, batch_size=1000):
    """Calculate mean and std of CIFAR-10 dataset"""
    print("Calculating dataset statistics...")
    
    # Load CIFAR-10 without normalization
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(
        root=data_path, 
        train=True, 
        transform=transform, 
        download=False  # Already downloaded
    )
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Calculate mean
    mean = torch.zeros(3)
    for images, _ in tqdm(train_loader, desc="Computing mean"):
        for i in range(3):  # RGB channels
            mean[i] += images[:, i, :, :].mean()
    mean /= len(train_loader)
    
    # Calculate std
    std = torch.zeros(3)
    for images, _ in tqdm(train_loader, desc="Computing std"):
        for i in range(3):
            std[i] += ((images[:, i, :, :] - mean[i]) ** 2).mean()
    std = torch.sqrt(std / len(train_loader))
    
    return mean.tolist(), std.tolist()

def download_cifar10(data_path, compute_stats=True):
    """
    Download CIFAR-10 dataset to the specified path and optionally compute normalization stats
    
    Args:
        data_path: Path where to download the dataset
        compute_stats: Whether to compute mean and std statistics
    
    Returns:
        dict: Contains success status and normalization statistics if computed
    """
    # Create the data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Path for statistics file
    stats_file = os.path.join(data_path, 'cifar10_stats.json')
    
    try:
        # Download both training and test datasets
        print(f"Downloading CIFAR-10 to {data_path}...")
        
        # Download training data
        train_dataset = datasets.CIFAR10(
            root=data_path, 
            train=True, 
            download=True
        )
        
        # Download test data
        test_dataset = datasets.CIFAR10(
            root=data_path, 
            train=False, 
            download=True
        )
        
        print(f"Successfully downloaded CIFAR-10 to {data_path}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Compute statistics if requested
        if compute_stats:
            if os.path.exists(stats_file):
                print(f"Loading existing statistics from {stats_file}")
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                mean = stats['mean']
                std = stats['std']
            else:
                print("Computing normalization statistics...")
                mean, std = calculate_dataset_stats(data_path)
                
                # Save statistics
                stats = {
                    'mean': mean,
                    'std': std,
                    'dataset': 'CIFAR-10',
                    'num_training_samples': len(train_dataset)
                }
                
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print(f"Statistics saved to {stats_file}")
            
            print(f"Mean: {mean}")
            print(f"Std: {std}")
            
            return {
                'success': True,
                'mean': mean,
                'std': std,
                'stats_file': stats_file
            }
        
        return {'success': True}
        
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")
        return {'success': False, 'error': str(e)}

def load_cifar10_with_stats(data_path, batch_size=128, num_workers=2, val_ratio=0.1, random_seed=42):
    """
    Load CIFAR-10 dataset with train/val/test split
    
    Args:
        data_path: Path to the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        val_ratio: Ratio of training data to use for validation (default: 0.1)
        random_seed: Random seed for train/val split reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, stats)
    """
    import os
    import json
    
    # Check if data already exists
    cifar_dir = os.path.join(data_path, 'cifar-10-batches-py')
    stats_file = os.path.join(data_path, 'cifar10_stats.json')
    
    # Load or compute stats
    if os.path.exists(stats_file):
        print(f"Loading existing statistics from {stats_file}")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        mean = stats['mean']
        std = stats['std']
    else:
        print("Computing normalization statistics...")
        # Only download/compute if needed
        result = download_cifar10(data_path, compute_stats=True)
        if not result['success']:
            raise RuntimeError(f"Failed to download CIFAR-10: {result.get('error', 'Unknown error')}")
        mean = result['mean']
        std = result['std']
    
    # Create transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load datasets - download=True will only download if data doesn't exist
    train_dataset = datasets.CIFAR10(
        root=data_path, 
        train=True, 
        transform=transform_train,
        download=True  # Only downloads if needed
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_path, 
        train=True, 
        transform=transform_test,
        download=False  # Already checked above
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_path, 
        train=False, 
        transform=transform_test,
        download=True  # Only downloads if needed
    )
    
    # Create train/val split
    total_size = len(train_dataset)
    indices = list(range(total_size))
    val_size = int(total_size * val_ratio)
    
    # Shuffle indices for random split
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Prepare stats
    stats = {
        'mean': mean,
        'std': std,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_dataset)
    }
    
    print(f"Dataset split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, stats

def verify_stats(data_path='./data'):
    """Verify that computed stats match the standard CIFAR-10 stats"""
    result = download_cifar10(data_path, compute_stats=True)
    
    if not result['success']:
        print("Failed to download or compute stats")
        return
    
    computed_mean = result['mean']
    computed_std = result['std']
    
    standard_mean = [0.4914, 0.4822, 0.4465]
    standard_std = [0.2023, 0.1994, 0.2010]
    
    print("\nComparison with standard CIFAR-10 statistics:")
    print("Channel | Computed Mean | Standard Mean | Difference")
    print("-" * 55)
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        diff = abs(computed_mean[i] - standard_mean[i])
        print(f"{channel:7} | {computed_mean[i]:.4f}       | {standard_mean[i]:.4f}       | {diff:.6f}")
    
    print("\nChannel | Computed Std  | Standard Std  | Difference")
    print("-" * 55)
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        diff = abs(computed_std[i] - standard_std[i])
        print(f"{channel:7} | {computed_std[i]:.4f}       | {standard_std[i]:.4f}       | {diff:.6f}")

def plot_confusion_matrix(cm, class_names, title, filename=None, figsize=(10, 8)):
    """Plot a confusion matrix with labels"""
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def get_model_predictions(model, dataloader, device):
    """
    Get predictions from a standard model (non-MoE)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions from model
            logits = model(images)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def create_confusion_matrix_standard(model, dataloader, device, output_dir, class_names):
    """
    Create confusion matrix for a standard model (non-MoE)
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for validation data
        device: Device to run computations on
        output_dir: Directory to save outputs
        class_names: List of class names
        
    Returns:
        predictions: Model predictions
        true_labels: Ground truth labels
        accuracy: Model accuracy
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    predictions, true_labels = get_model_predictions(model, dataloader, device)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
    
    # Plot confusion matrix
    title = f'Model Confusion Matrix (Accuracy: {accuracy:.2f}%)'
    filename = os.path.join(output_dir, 'confusion_matrix.png')
    
    plot_confusion_matrix(cm, class_names, title, filename)
    print(f"Model accuracy: {accuracy:.2f}%")
    
    # Save results to JSON
    results = {
        "accuracy": round(accuracy, 2),
        "confusion_matrix": cm.tolist(),
        "class_accuracies": {}
    }
    
    # Calculate per-class accuracy
    for i, class_name in enumerate(class_names):
        class_mask = (true_labels == i)
        if np.sum(class_mask) > 0:
            class_correct = np.sum((predictions == i) & class_mask)
            class_total = np.sum(class_mask)
            class_accuracy = class_correct / class_total * 100
            results["class_accuracies"][class_name] = round(class_accuracy, 2)
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return predictions, true_labels, accuracy

def plot_training_history(info, plots_dir):
    """
    Plot training and validation loss curves
    
    Args:
        info: Dictionary containing training history from train_model()
        plots_dir: Directory to save plots
        
    Returns:
        Path to loss plot
    """
    import matplotlib.pyplot as plt
    import os
    
    # Extract data, handling potential None values
    train_losses = info['tr']['xent']
    val_losses = info['va']['xent']
    
    # Remove any None values from the beginning of train losses
    start_idx = 0
    while start_idx < len(train_losses) and train_losses[start_idx] is None:
        start_idx += 1
    
    train_losses = train_losses[start_idx:]
    
    # Check if validation losses need the same treatment
    val_start_idx = 0
    while val_start_idx < len(val_losses) and val_losses[val_start_idx] is None:
        val_start_idx += 1
    
    val_losses = val_losses[val_start_idx:]
    
    # Take the minimum length to ensure matching dimensions
    min_length = min(len(train_losses), len(val_losses))
    train_losses = train_losses[:min_length]
    val_losses = val_losses[:min_length]
    
    # Create epochs based on the final length
    epochs = range(1, min_length + 1)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    
    # Mark the best epoch
    best_epoch = info['best_epoch']
    best_val_loss = info['best_va_loss']
    
    # Adjust best_epoch if we skipped initial values
    adjusted_best_epoch = best_epoch - start_idx
    if adjusted_best_epoch > 0 and adjusted_best_epoch <= min_length:
        plt.scatter([adjusted_best_epoch], [best_val_loss], color='green', s=100, marker='*', 
                    label=f'Best Model (Epoch {best_epoch})', zorder=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(plots_dir, 'training_validation_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training/Validation loss plot saved to: {loss_plot_path}")
    
    return loss_plot_path

## 

def get_expert_predictions_corrected(model, dataloader, device):
    """
    Get predictions from each expert using the correct architecture with skip connections
    """
    model.eval()
    
    num_experts = model.num_experts
    expert_preds = [[] for _ in range(num_experts)]
    full_model_preds = []
    true_labels = []
    
    # Debug: Count predictions per class for each expert
    expert_class_counts = [{} for _ in range(num_experts)]
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Get predictions from each expert WITH skip connection
            for i in range(num_experts):
                # Get expert features
                _, expert_features = model.experts[i](images)
                
                # Get skip features if skip connection is used
                if model.use_skip:
                    skip_features = model.skip_path(images)
                    combined_features = torch.cat([expert_features, skip_features], dim=1)
                    
                    # Use the actual final classifier
                    expert_logits = model.final_fc(combined_features)
                else:
                    # If no skip connection, use expert's own logits
                    expert_logits, _ = model.experts[i](images)
                
                expert_pred = expert_logits.argmax(dim=1)
                expert_preds[i].extend(expert_pred.cpu().numpy())
                
                # Debug: Count predictions per class
                for pred in expert_pred.cpu().numpy():
                    expert_class_counts[i][pred] = expert_class_counts[i].get(pred, 0) + 1
            
            # Get predictions from full model
            full_logits = model(images)
            full_pred = full_logits.argmax(dim=1)
            full_model_preds.extend(full_pred.cpu().numpy())
            
            # Store true labels
            true_labels.extend(labels.cpu().numpy())
            
            # Debug print every 10 batches
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Debug: Print prediction distribution for each expert
    for i in range(num_experts):
        print(f"\nExpert {i+1} prediction distribution:")
        for class_id, count in sorted(expert_class_counts[i].items()):
            print(f"  Class {class_id}: {count} predictions")
    
    return expert_preds, np.array(full_model_preds), np.array(true_labels)

def create_expert_analysis(expert_preds, true_labels, class_names, output_dir):
    """Create comprehensive expert analysis and visualizations"""
    num_experts = len(expert_preds)
    
    # Convert expert predictions to numpy arrays if they aren't already
    expert_preds_arrays = []
    for i in range(num_experts):
        if isinstance(expert_preds[i], list):
            expert_preds_arrays.append(np.array(expert_preds[i]))
        else:
            expert_preds_arrays.append(expert_preds[i])
    
    # Convert true_labels to numpy array if it isn't already
    if isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    
    # Debug: Check shapes
    print(f"True labels shape: {true_labels.shape}")
    for i, expert_pred in enumerate(expert_preds_arrays):
        print(f"Expert {i+1} predictions shape: {expert_pred.shape}")
        assert true_labels.shape[0] == expert_pred.shape[0], f"Mismatch in number of predictions for expert {i+1}!"
    
    # Create confusion matrix for each expert
    for i in range(num_experts):
        cm = confusion_matrix(true_labels, expert_preds_arrays[i])
        accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
        
        title = f'Expert {i+1} Confusion Matrix (Accuracy: {accuracy:.2f}%)'
        filename = os.path.join(output_dir, f'expert_{i+1}_confusion_matrix.png')
        
        plot_confusion_matrix(cm, class_names, title, filename)
        print(f"    Expert {i+1} accuracy: {accuracy:.2f}%")
    
    # Calculate per-class accuracy for each expert
    expert_accuracies = np.zeros((num_experts, len(class_names)))
    
    for i in range(num_experts):
        expert_preds_array = expert_preds_arrays[i]
        
        for j, class_name in enumerate(class_names):
            class_mask = (true_labels == j)
            if np.sum(class_mask) > 0:
                correct = np.sum((expert_preds_array == j) & class_mask)
                total = np.sum(class_mask)
                expert_accuracies[i, j] = correct / total * 100
                
                # Debug print
                print(f"Expert {i+1}, Class {class_name}: {correct}/{total} = {expert_accuracies[i, j]:.2f}%")
            else:
                print(f"Warning: No true labels for class {class_name}")
    
    # Create heatmap of expert specializations
    plt.figure(figsize=(12, 8))
    sns.heatmap(expert_accuracies, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=class_names, 
                yticklabels=[f'Expert {i+1}' for i in range(num_experts)])
    
    plt.title('Expert Specialization: Per-Class Accuracy (%)')
    plt.xlabel('Class')
    plt.ylabel('Expert')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'expert_specialization_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_expert_visualization(expert_preds, true_labels, class_names, output_dir):
    """Create a combined visualization showing what each expert specializes in"""
    num_experts = len(expert_preds)
    
    # Convert to numpy arrays
    expert_preds_arrays = []
    for i in range(num_experts):
        if isinstance(expert_preds[i], list):
            expert_preds_arrays.append(np.array(expert_preds[i]))
        else:
            expert_preds_arrays.append(expert_preds[i])
    
    # Debug: Verify we have predictions
    for i, expert_pred in enumerate(expert_preds_arrays):
        print(f"Expert {i+1} has {len(expert_pred)} predictions")
        if len(expert_pred) == 0:
            print(f"Warning: Expert {i+1} has no predictions!")
    
    # Calculate per-class accuracy for each expert
    expert_accuracies = np.zeros((num_experts, len(class_names)))
    
    for i in range(num_experts):
        expert_preds_array = expert_preds_arrays[i]
        
        # Check if expert makes any predictions
        if len(expert_preds_array) == 0:
            print(f"Warning: Expert {i+1} has no predictions!")
            continue
            
        for j, class_name in enumerate(class_names):
            class_mask = (true_labels == j)
            if np.sum(class_mask) > 0:
                correct = np.sum((expert_preds_array == j) & class_mask)
                total = np.sum(class_mask)
                expert_accuracies[i, j] = correct / total * 100
                
                # Check if expert never predicts this class
                expert_predicts_class = np.sum(expert_preds_array == j)
                if expert_predicts_class == 0:
                    print(f"Warning: Expert {i+1} never predicts class {class_name}")
                else:
                    print(f"Expert {i+1} predicts class {class_name} {expert_predicts_class} times")
    
    # Create heatmap of expert specializations
    plt.figure(figsize=(12, 8))
    sns.heatmap(expert_accuracies, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=class_names, 
                yticklabels=[f'Expert {i+1}' for i in range(num_experts)])
    
    plt.title('Expert Specialization: Per-Class Accuracy (%)')
    plt.xlabel('Class')
    plt.ylabel('Expert')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'expert_specialization_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Expert specialization heatmap saved to {filename}")

    # Create comprehensive expert analysis JSON
    expert_analysis = {
        "expert_accuracies": {
            "description": "Per-class accuracy for each expert",
            "data": {}
        },
        "best_expert_per_class": {
            "description": "Which expert performs best for each class",
            "data": {}
        },
        "expert_rankings_by_class": {
            "description": "Experts ranked by accuracy for each class",
            "data": {}
        },
        "class_rankings_by_expert": {
            "description": "Classes ranked by accuracy for each expert",
            "data": {}
        },
        "overall_expert_accuracy": {
            "description": "Overall accuracy for each expert across all classes",
            "data": {}
        }
    }

    # Fill in expert accuracies
    for i in range(num_experts):
        expert_name = f"expert_{i+1}"
        expert_analysis["expert_accuracies"]["data"][expert_name] = {}
        
        for j, class_name in enumerate(class_names):
            accuracy = expert_accuracies[i, j]
            expert_analysis["expert_accuracies"]["data"][expert_name][class_name] = round(accuracy, 2)
        
        # Overall accuracy for this expert
        overall_accuracy = np.mean(expert_accuracies[i, :])
        expert_analysis["overall_expert_accuracy"]["data"][expert_name] = round(overall_accuracy, 2)

    # Determine best expert per class
    best_expert_per_class = np.argmax(expert_accuracies, axis=0)
    for j, class_name in enumerate(class_names):
        best_expert_idx = best_expert_per_class[j]
        best_accuracy = expert_accuracies[best_expert_idx, j]
        expert_analysis["best_expert_per_class"]["data"][class_name] = {
            "expert": f"expert_{best_expert_idx+1}",
            "accuracy": round(best_accuracy, 2)
        }

    # Expert rankings for each class
    for j, class_name in enumerate(class_names):
        class_accuracies = expert_accuracies[:, j]
        ranking_indices = np.argsort(-class_accuracies)  # Sort descending
        rankings = []
        for idx in ranking_indices:
            rankings.append({
                "expert": f"expert_{idx+1}",
                "accuracy": round(class_accuracies[idx], 2)
            })
        expert_analysis["expert_rankings_by_class"]["data"][class_name] = rankings

    # Class rankings for each expert
    for i in range(num_experts):
        expert_name = f"expert_{i+1}"
        expert_accuracies_list = expert_accuracies[i, :]
        
        # Get indices sorted by accuracy
        sorted_indices = np.argsort(-expert_accuracies_list)
        
        rankings = []
        for idx in sorted_indices:
            rankings.append({
                "class": class_names[idx],
                "accuracy": round(expert_accuracies_list[idx], 2)
            })
        expert_analysis["class_rankings_by_expert"]["data"][expert_name] = rankings

    # Save expert analysis to JSON
    json_filename = os.path.join(output_dir, 'expert_analysis.json')
    with open(json_filename, 'w') as f:
        json.dump(expert_analysis, f, indent=2)

    print(f"Expert analysis saved to {json_filename}")

    # Create raw_data directory
    raw_data_dir = os.path.join(output_dir, 'raw_data')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Save raw accuracy matrix to raw_data directory
    np.save(os.path.join(raw_data_dir, 'expert_accuracies.npy'), expert_accuracies)

def create_expert_confusion_matrices(model, data_loader, device, plots_dir, class_names):
    """
    Create confusion matrices and comprehensive analysis for each expert in the model.
    """
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get predictions from each expert using the correct architecture
    expert_preds, full_model_preds, true_labels = get_expert_predictions_corrected(
        model, data_loader, device
    )
    
    # Debug: Print summary information
    print(f"\nSummary:")
    print(f"Number of experts: {len(expert_preds)}")
    print(f"Number of samples: {len(true_labels)}")
    print(f"Classes: {class_names}")
    print(f"Unique true labels: {np.unique(true_labels)}")
    
    # Create confusion matrix for each expert and analysis
    create_expert_analysis(expert_preds, true_labels, class_names, plots_dir)
    
    # Create confusion matrix for the full model
    cm_full = confusion_matrix(true_labels, full_model_preds)
    accuracy_full = np.sum(np.diag(cm_full)) / np.sum(cm_full) * 100
    
    title_full = f'Full Model Confusion Matrix (Accuracy: {accuracy_full:.2f}%)'
    filename_full = os.path.join(plots_dir, 'full_model_confusion_matrix.png')
    
    plot_confusion_matrix(cm_full, class_names, title_full, filename_full)
    print(f"Full model accuracy: {accuracy_full:.2f}%")
    
    # Create combined visualization showing expert specializations
    create_combined_expert_visualization(expert_preds, true_labels, class_names, plots_dir)
    
    # Create analysis_result.json
    analysis_result = {
        "full_model_accuracy": round(accuracy_full, 2),
        "expert_accuracies": {},
        "confusion_matrices": {
            "full_model": cm_full.tolist()
        }
    }
    
    # Add individual expert accuracies and confusion matrices
    for i, expert_pred in enumerate(expert_preds):
        expert_pred_array = np.array(expert_pred)
        cm = confusion_matrix(true_labels, expert_pred_array)
        accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
        
        analysis_result["expert_accuracies"][f"expert_{i+1}"] = round(accuracy, 2)
        analysis_result["confusion_matrices"][f"expert_{i+1}"] = cm.tolist()
    
    # Save analysis results
    with open(os.path.join(plots_dir, 'analysis_result.json'), 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"Analysis results saved to {os.path.join(plots_dir, 'analysis_result.json')}")
    
    return expert_preds, full_model_preds, true_labels