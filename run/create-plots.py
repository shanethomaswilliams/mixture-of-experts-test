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

# Add the parent directory to the path to find the src module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.models import AdaptiveMoEWithSkip
    from src.utils import load_cifar10_with_stats
except ImportError:
    # Try alternate import path
    sys.path.append(os.path.join(parent_dir, 'src'))
    try:
        from models import AdaptiveMoEWithSkip
        from utils import load_cifar10_with_stats
    except ImportError:
        # If that fails, try importing directly
        try:
            from models import AdaptiveMoEWithSkip
            from utils import load_cifar10_with_stats
        except ImportError:
            print("Error: Could not import required modules. Please check your directory structure.")
            print(f"Current directory: {os.getcwd()}")
            print(f"Script directory: {current_dir}")
            print(f"Parent directory: {parent_dir}")
            print(f"Python path: {sys.path}")
            sys.exit(1)

# You'll need to import your model classes and helper functions
# from your_model_file import AdaptiveMoEWithSkip
# from your_data_utils import load_cifar10_with_stats

def parse_experiment_name(exp_name):
    """Parse experiment directory name to extract parameters"""
    pattern = r"num-experts_(\d+)_topk_(\d+)_lr_([\d\.]+)"
    match = re.match(pattern, exp_name)
    if match:
        return {
            'num_experts': int(match.group(1)),
            'top_k': int(match.group(2)),
            'learning_rate': float(match.group(3))
        }
    return None

def get_expert_predictions_corrected(model, dataloader, device):
    """
    Get predictions from each expert using the correct architecture with skip connections
    """
    model.eval()
    
    num_experts = model.num_experts
    expert_preds = [[] for _ in range(num_experts)]
    full_model_preds = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
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
            
            # Get predictions from full model
            full_logits = model(images)
            full_pred = full_logits.argmax(dim=1)
            full_model_preds.extend(full_pred.cpu().numpy())
            
            # Store true labels
            true_labels.extend(labels.cpu().numpy())
    
    return expert_preds, np.array(full_model_preds), np.array(true_labels)

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

def create_expert_analysis(expert_preds, true_labels, class_names, output_dir):
    """Create comprehensive expert analysis and visualizations"""
    num_experts = len(expert_preds)
    
    # Create confusion matrix for each expert
    for i in range(num_experts):
        cm = confusion_matrix(true_labels, expert_preds[i])
        accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
        
        title = f'Expert {i+1} Confusion Matrix (Accuracy: {accuracy:.2f}%)'
        filename = os.path.join(output_dir, f'expert_{i+1}_confusion_matrix.png')
        
        plot_confusion_matrix(cm, class_names, title, filename)
        print(f"    Expert {i+1} accuracy: {accuracy:.2f}%")
    
    # Calculate per-class accuracy for each expert
    expert_accuracies = np.zeros((num_experts, len(class_names)))
    
    for i in range(num_experts):
        # Convert expert predictions to numpy array
        expert_preds_array = np.array(expert_preds[i])
        
        for j, class_name in enumerate(class_names):
            class_mask = (true_labels == j)
            if np.sum(class_mask) > 0:
                correct = np.sum((expert_preds_array == j) & class_mask)
                total = np.sum(class_mask)
                expert_accuracies[i, j] = correct / total * 100
    
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

def verify_model_architecture(model, state_dict):
    """Verify that the model architecture matches the saved state dict"""
    model_state = model.state_dict()
    
    # Check if all keys match
    model_keys = set(model_state.keys())
    state_dict_keys = set(state_dict.keys())
    
    missing_keys = state_dict_keys - model_keys
    unexpected_keys = model_keys - state_dict_keys
    
    if missing_keys:
        print(f"  WARNING: Missing keys in model: {missing_keys}")
        return False
    
    if unexpected_keys:
        print(f"  WARNING: Unexpected keys in model: {unexpected_keys}")
        return False
    
    # Check if shapes match
    shape_mismatches = []
    for key in model_state.keys():
        if key in state_dict:
            model_shape = model_state[key].shape
            saved_shape = state_dict[key].shape
            if model_shape != saved_shape:
                shape_mismatches.append(
                    f"    {key}: model shape {model_shape} vs saved shape {saved_shape}"
                )
    
    if shape_mismatches:
        print(f"  ERROR: Shape mismatches found:")
        for mismatch in shape_mismatches:
            print(mismatch)
        return False
    
    print("  ✓ Model architecture verified - all keys and shapes match")
    return True

def process_all_experiments(experiments_dir, data_path, device):
    """Process all experiments in the experiments directory"""
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load dataloaders once
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader, stats = load_cifar10_with_stats(data_path)
    
    # Process each experiment directory
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        
        if not os.path.isdir(exp_path):
            continue
            
        # Parse experiment parameters
        params = parse_experiment_name(exp_dir)
        if not params:
            print(f"Skipping {exp_dir} - invalid experiment name format")
            continue
            
        print(f"\nProcessing experiment: {exp_dir}")
        print(f"  Parameters: {params}")
        
        # Create paths
        models_dir = os.path.join(exp_path, 'models')
        plots_dir = os.path.join(exp_path, 'plots')
        
        # Find the most recent model file
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            print(f"  No model files found in {models_dir}")
            continue
            
        # Sort by modification time and get the most recent
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
        model_file = model_files[-1]
        model_path = os.path.join(models_dir, model_file)
        
        print(f"  Loading model from {model_file}")
        
        # Load model
        try:
            # Create model with correct parameters
            model = AdaptiveMoEWithSkip(
                num_experts=params['num_experts'],
                num_classes=10,
                use_skip=True,
                top_k=params['top_k']
            )
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # Verify architecture matches
            if not verify_model_architecture(model, state_dict):
                print(f"  ERROR: Architecture mismatch for {exp_dir}")
                continue
                
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            # Print model info for verification
            print(f"  Model info:")
            print(f"    Number of experts: {model.num_experts}")
            print(f"    Top-k: {model.top_k}")
            print(f"    Use skip: {model.use_skip}")
            print(f"    Base width: {model.base_width}")
            print(f"    Num layers: {model.num_layers}")
            print(f"    Num blocks: {model.num_blocks}")
            
            print("  Generating predictions and visualizations...")
            
            # Get predictions using the corrected method
            expert_preds, full_model_preds, true_labels = get_expert_predictions_corrected(
                model, test_loader, device
            )
            
            # Create visualizations
            create_expert_analysis(expert_preds, true_labels, class_names, plots_dir)
            
            # Create confusion matrix for full model
            cm_full = confusion_matrix(true_labels, full_model_preds)
            accuracy_full = np.sum(np.diag(cm_full)) / np.sum(cm_full) * 100
            
            title_full = f'Full MoE Model Confusion Matrix (Accuracy: {accuracy_full:.2f}%)'
            filename_full = os.path.join(plots_dir, 'full_model_confusion_matrix.png')
            
            plot_confusion_matrix(cm_full, class_names, title_full, filename_full)
            print(f"  Full model accuracy: {accuracy_full:.2f}%")
            
            # Save additional analysis in JSON format
            analysis_results = {
                'experiment_params': params,
                'model_architecture': {
                    'base_width': model.base_width,
                    'num_layers': model.num_layers,
                    'num_blocks': model.num_blocks,
                    'num_experts': model.num_experts,
                    'top_k': model.top_k,
                    'use_skip': model.use_skip
                },
                'full_model_accuracy': accuracy_full,
                'expert_accuracies': {
                    f'expert_{i+1}': float(np.mean([
                        np.sum((np.array(expert_preds[i]) == j) & (true_labels == j)) / 
                        np.sum(true_labels == j) * 100
                        for j in range(10)
                    ]))
                    for i in range(params['num_experts'])
                }
            }
            
            # Save analysis results
            json_path = os.path.join(plots_dir, 'analysis_results.json')
            with open(json_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            print(f"  Visualizations saved to {plots_dir}")
            
        except Exception as e:
            print(f"  ERROR processing {exp_dir}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def main():
    # Configuration
    experiments_dir = "../experiments"
    data_path = "../data"  # Path to CIFAR-10 data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\
    
    print(f"Using device: {device}")
    
    # Process all experiments
    process_all_experiments(experiments_dir, data_path, device)
    
    print("\nAll experiments processed successfully!")

if __name__ == "__main__":
    main()