import argparse
import json
import logging
import os
import sys
import torch

sys.path.append("../src/")
from sklearn import metrics
import numpy as np
import math
import losses
from models import AdaptiveMoEWithSkip, ResNet18, TraditionalMoEWithSkip
from src.utils import create_expert_confusion_matrices, train_model, evaluate_model, load_cifar10_with_stats, plot_training_history, create_confusion_matrix_standard
import torch
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Train MoE model on CIFAR10 dataset")

def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')
    
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

# CIFAR10 PARAMETERS
parser.add_argument("--dataset_name", type=str, default="cifar10", 
                    help="Dataset to use (cifar10 or cifar100)")
parser.add_argument("--data_dir", type=str, default="./data", 
                    help="Directory containing the dataset")
parser.add_argument("--batch_size", type=int, default=128, 
                    help="Batch size for training and evaluation")
parser.add_argument("--num_classes", type=int, default=100,
                    help="Number of output classes")
parser.add_argument("--val_ratio", type=float, default=0.1,
                    help="Train/Validation ratio for CIFAR10")

# MODEL PARAMETERS
parser.add_argument("--num_experts", type=int, default=4, 
                    help="Number of experts for MoE model")
parser.add_argument("--use_skip", type=str2bool, default=True, 
                    help="Allowing skip layers to prevent expert vanishing gradients")
parser.add_argument("--top_k", type=int, default=1, 
                    help="Number of experts allowed to be used")

# TRAINING PARAMETERS
parser.add_argument("--learning_rate", type=float, default=0.01, 
                    help="Learning rate for training")
parser.add_argument("--l2pen_mag", type=float, default=0.0,
                    help="Weight for the L2 regularization")
parser.add_argument("--n_epochs", type=int, default=20000, 
                    help="Maximum number of training epochs")
parser.add_argument("--do_early_stopping", type=str2bool, default=True,
                    help="Enable early stopping")
parser.add_argument("--patience", type=int, default=15, 
                    help="Patience for early stopping")
parser.add_argument('--switch_balance_weight', type=float, default=0.1,
                    help='Weight for switch-style load balancing loss that encourages balanced expert selection (default: 0.1)')
parser.add_argument('--importance_weight', type=float, default=0.05,
                    help='Weight for importance loss with entropy penalty to balance expert usage while avoiding uniform distributions (default: 0.05)')
parser.add_argument('--topk_balance_weight', type=float, default=0.1,
                    help='Weight for top-k specific load balancing loss when using top-k > 1 routing (default: 0.1)')
parser.add_argument('--mutual_info_weight', type=float, default=0.05,
                    help='Weight for mutual information loss that encourages experts to specialize on different classes (default: 0.05)')
parser.add_argument('--entropy_weight', type=float, default=0.01,
                    help='Weight for entropy regularization to encourage diverse routing decisions (default: 0.01)')


# OUTPUT/SAVING PARAMETERS
parser.add_argument("--run_label", type=str, default="default_run", 
                    help="Label for this training run")
parser.add_argument("--output_dir", type=str, default="./results", 
                    help="Directory to save outputs")
parser.add_argument("--model_filename", type=str, default="model.pth",
                    help="Filename for saved model")
parser.add_argument("--preds_filename", type=str, default="preds.pt",
                    help="Filename for saved predictions")
parser.add_argument("--fig_filename", type=str, default="plot.png",
                    help="Filename for saved figures")
parser.add_argument("--results_file", type=str, default="results.json",
                    help="Filename for saved results")

args = parser.parse_args()

# SET DEVICE TO GPU IF AVAILABLE
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# LOAD CIFAR10 
if args.dataset_name.lower() == "cifar10":
    train_loader, val_loader, test_loader, stats = load_cifar10_with_stats(args.data_dir)
else:
    raise ValueError(f"Dataset '{args.dataset_name}' has not been implemented. Only 'cifar10' is currently supported.")

# CREATE FINAL DIRECTORIES FOR SAVING INFORMATION
os.makedirs(args.output_dir, exist_ok=True)
model_dir = os.path.dirname(args.model_filename)
preds_dir = os.path.dirname(args.preds_filename)
plots_dir = os.path.dirname(args.fig_filename)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(preds_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# PRINT DATASET AND TRAINING INFORMATION
print("=========================PRE-TRAINING SUMMARY=========================")
print(f"DATASET CONFIGURATION:")
print(f"  Dataset: {args.dataset_name}")
print(f"  Data directory: {args.data_dir}")
print(f"  Number of classes: {args.num_classes}")
print(f"  Batch size: {args.batch_size}")
print(f"  Validation ratio: {args.val_ratio}")

print(f"\nMIXTURE OF EXPERTS (MoE) CONFIGURATION:")
print(f"  Number of experts: {args.num_experts}")
print(f"  Top-K experts active: {args.top_k}")
print(f"  Use skip connections: {args.use_skip}")

if args.num_experts > 1:
    print(f"\nMoE LOSS WEIGHTS:")
    print(f"  Switch balance weight: {args.switch_balance_weight}")
    print(f"  Importance weight: {args.importance_weight}")
    print(f"  Top-K balance weight: {args.topk_balance_weight}")
    print(f"  Mutual information weight: {args.mutual_info_weight}")
    print(f"  Entropy weight: {args.entropy_weight}")

print(f"\nTRAINING CONFIGURATION:")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Maximum epochs: {args.n_epochs}")
print(f"  Early stopping: {args.do_early_stopping}")
if args.do_early_stopping:
    print(f"  Early stopping patience: {args.patience}")

print(f"\nOUTPUT CONFIGURATION:")
print(f"  Run label: {args.run_label}")
print(f"  Output directory: {args.output_dir}")
print(f"  Model filename: {args.model_filename}")
print(f"  Predictions filename: {args.preds_filename}")
print(f"  Figure filename: {args.fig_filename}")
print(f"  Results filename: {args.results_file}")

print(f"\nSYSTEM CONFIGURATION:")
print(f"  Device: {device}")

config_summary = {
    "dataset_configuration": {
        "dataset": args.dataset_name,
        "data_directory": args.data_dir,
        "number_of_classes": args.num_classes,
        "batch_size": args.batch_size,
        "validation_ratio": args.val_ratio
    },
    "mixture_of_experts_configuration": {
        "number_of_experts": args.num_experts,
        "top_k_experts_active": args.top_k,
        "use_skip_connections": args.use_skip
    },
    "training_configuration": {
        "learning_rate": args.learning_rate,
        "maximum_epochs": args.n_epochs,
        "early_stopping": args.do_early_stopping,
        "early_stopping_patience": args.patience if args.do_early_stopping else None
    },
    "output_configuration": {
        "run_label": args.run_label,
        "output_directory": args.output_dir,
        "model_filename": args.model_filename,
        "predictions_filename": args.preds_filename,
        "figure_filename": args.fig_filename,
        "results_filename": args.results_file
    },
    "system_configuration": {
        "device": device
    }
}

if args.num_experts > 1:
    config_summary["moe_loss_weights"] = {
        "switch_balance_weight": args.switch_balance_weight,
        "importance_weight": args.importance_weight,
        "topk_balance_weight": args.topk_balance_weight,
        "mutual_info_weight": args.mutual_info_weight,
        "entropy_weight": args.entropy_weight
    }

initialization_json_path = os.path.join(args.output_dir, "initialization_summary.json")
with open(initialization_json_path, 'w') as f:
    json.dump(config_summary, f, indent=4)
print("=====================================================================\n")

print("=========================MODEL BREAKDOWN=========================")
# INITIALIZE MODEL
try:
    if args.num_experts == 1:
        model = ResNet18()
    else:
        model = AdaptiveMoEWithSkip(num_experts=args.num_experts, num_classes=args.num_classes, use_skip=args.use_skip, top_k=args.top_k)
except TypeError:
    raise TypeError(f"Issue ")
print(f"MODEL INITIALIZED: {model.__class__.__name__}\n")
print(model)
print("=================================================================\n\n")

# TRAIN MODEL
print("=========================TRAINING MODEL==========================")
trained_model, info = train_model(
    model,
    device,
    train_loader,
    val_loader,
    losses,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    lr=args.learning_rate,
    l2pen_mag=args.l2pen_mag,
    do_early_stopping=args.do_early_stopping,
    model_filename=args.model_filename,
    n_epochs_without_va_improve_before_early_stop=args.patience,
    switch_balance_weight=args.switch_balance_weight, 
    importance_weight=args.importance_weight,
    topk_balance_weight=args.topk_balance_weight, 
    mutual_info_weight=args.mutual_info_weight,
    entropy_weight=args.entropy_weight
)
training_info_path = os.path.join(model_dir, "training_info.json")
info = clean_for_json(info)
with open(training_info_path, 'w') as f:
    json.dump(info, f, indent=4)
print(f"TRAINING INFO SAVED TO: {training_info_path}")
print("=================================================================\n\n")


print("========================EVALUATING MODEL=========================")
# EVALUATE TRAIN DATASET
train_accuracy, train_outputs, train_labels, train_predictions = evaluate_model(trained_model, train_loader, device)
print(f"TRAIN ACCURACY: {train_accuracy:.4f}")
train_preds_path = os.path.join(preds_dir, "train_preds.pt")
torch.save({
    'outputs': train_outputs,
    'labels': train_labels,
    'predictions': train_predictions,
    'accuracy': train_accuracy
}, train_preds_path)
print(f"TRAIN PREDICTIONS SAVED TO: {train_preds_path}\n")

# EVALUATE VALIDATION DATASET
validation_accuracy, validation_outputs, validation_labels, validation_predictions = evaluate_model(trained_model, val_loader, device)
print(f"VALIDATION ACCURACY: {validation_accuracy:.4f}")
validation_preds_path = os.path.join(preds_dir, "validation_preds.pt")
torch.save({
    'outputs': validation_outputs,
    'labels': validation_labels,
    'predictions': validation_predictions,
    'accuracy': validation_accuracy
}, validation_preds_path)
print(f"VALIDATION PREDICTIONS SAVED TO: {validation_preds_path}\n")

# EVALUATE ON TEST DATASET
test_accuracy, test_outputs, test_labels, test_predictions = evaluate_model(trained_model, test_loader, device)
print(f"TEST ACCURACY: {test_accuracy:.4f}")
test_preds_path = os.path.join(preds_dir, "test_preds.pt")
torch.save({
    'outputs': test_outputs,
    'labels': test_labels,
    'predictions': test_predictions,
    'accuracy': test_accuracy
}, test_preds_path)
print(f"TEST PREDICTIONS SAVED TO: {test_preds_path}\n")
print("=================================================================\n\n")

print("========================CREATING FIGURES=========================")

if hasattr(test_loader.dataset, 'dataset'):
    class_names = test_loader.dataset.dataset.classes
else:
    class_names = test_loader.dataset.classes

if not class_names:
    raise AttributeError("Could not find Cifar10 labels")

print(f"Class names from dataset: {class_names}")

plot_training_history(info, plots_dir)

if args.num_experts == 1:
    print("\nCreating confusion matrix for test set...")
    create_confusion_matrix_standard(model, val_loader, device, plots_dir, class_names)
else:
    print("\nCreating confusion matrices for test set...")
    expert_preds, full_model_preds, true_labels = create_expert_confusion_matrices(
        model, test_loader, device, plots_dir, class_names=class_names
    )
print("=================================================================\n\n")

print("====================SAVING TRAINING SUMMARY=======================")
# STORE FINAL RESULTS
final_loss = info["best_va_loss"]
print(f"FINAL VALIDATION LOSS: {final_loss:.4f}")

# Create a comprehensive summary
print("====================SAVING TRAINING SUMMARY=======================")
# STORE FINAL RESULTS
final_loss = info["best_va_loss"]
print(f"FINAL VALIDATION LOSS: {final_loss:.4f}")

# Create a comprehensive summary
summary = {
    "dataset_configuration": {
        "dataset": args.dataset_name,
        "data_directory": args.data_dir,
        "number_of_classes": args.num_classes,
        "batch_size": args.batch_size,
        "validation_ratio": args.val_ratio
    },
    "mixture_of_experts_configuration": {
        "number_of_experts": args.num_experts,
        "top_k_experts_active": args.top_k,
        "use_skip_connections": args.use_skip
    },
    "training_configuration": {
        "learning_rate": args.learning_rate,
        "maximum_epochs": args.n_epochs,
        "early_stopping": args.do_early_stopping,
        "early_stopping_patience": args.patience if args.do_early_stopping else None,
        "actual_epochs_trained": info.get("epoch", 0)
    },
    "performance_metrics": {
        "final_validation_loss": final_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "best_validation_accuracy": info.get("best_va_acc", validation_accuracy)
    },
    "training_history": {
        "train_losses": info.get("train_losses", []),
        "val_losses": info.get("va_losses", []), 
        "train_accuracies": info.get("train_accs", []),
        "val_accuracies": info.get("va_accs", [])
    },
    "output_files": {
        "model_file": args.model_filename,
        "predictions_directory": preds_dir,
        "train_predictions": os.path.join(preds_dir, "train_preds.pt"),
        "validation_predictions": os.path.join(preds_dir, "validation_preds.pt"),
        "test_predictions": os.path.join(preds_dir, "test_preds.pt"),
        "figure_file": args.fig_filename,
        "results_file": args.results_file,
        "training_info_file": training_info_path
    },
    "system_configuration": {
        "device": device,
        "run_label": args.run_label
    },
    "timestamp": info.get("timestamp", None)
}

# Add MoE loss weights to the final summary if using multiple experts
if args.num_experts > 1:
    summary["moe_loss_weights"] = {
        "switch_balance_weight": args.switch_balance_weight,
        "importance_weight": args.importance_weight,
        "topk_balance_weight": args.topk_balance_weight,
        "mutual_info_weight": args.mutual_info_weight,
        "entropy_weight": args.entropy_weight
    }

with open(args.results_file, "w") as f:
    json.dump(summary, f, indent=4)

print(f"TRAINING SUMMARY SAVED TO: {args.results_file}")
print("\nFINAL PERFORMANCE METRICS:")
print(f"  Train Accuracy: {train_accuracy:.4f}")
print(f"  Validation Accuracy: {validation_accuracy:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Best Validation Loss: {final_loss:.4f}")
print(f"  Epochs Trained: {info['epochs'] - 1}/{args.n_epochs}")
if args.num_experts > 1:
    print("\nMoE LOSS WEIGHTS USED:")
    print(f"  Switch balance weight: {args.switch_balance_weight}")
    print(f"  Importance weight: {args.importance_weight}")
    print(f"  Top-K balance weight: {args.topk_balance_weight}")
    print(f"  Mutual information weight: {args.mutual_info_weight}")
    print(f"  Entropy weight: {args.entropy_weight}")
print("=================================================================\n\n")

print("TRAINING AND EVALUATION COMPLETE")