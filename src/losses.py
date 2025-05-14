import torch # type: ignore
import torch.nn # type: ignore
import numpy as np # type: ignore
import utils

def calc_xent_loss(logits, y, reduction='mean'):
    """Calculate standard cross-entropy loss using PyTorch"""
    return torch.nn.functional.cross_entropy(logits, y, reduction=reduction)

def calc_l2_penalty(model, l2pen_mag, batch_size):
    """Calculate L2 regularization penalty"""
    if l2pen_mag > 0:
        params = utils.flatten_params(model)
        l2_loss = l2pen_mag * torch.sum(params ** 2) / batch_size
        return l2_loss
    return 0.0

def calc_xent_loss_with_l2(logits_BC, y_B, model, l2pen_mag, batch_size, reduction='mean'):
    """Calculate cross-entropy loss with L2 regularization"""
    ce_loss = calc_xent_loss(logits_BC, y_B, reduction=reduction)
    l2_loss = calc_l2_penalty(model, l2pen_mag, batch_size)
    return ce_loss + l2_loss

def calc_switch_load_balancing_loss(model, epsilon=1e-8):
    """Encourage balanced assignment of samples to experts as their TOP choice"""
    if not hasattr(model, 'last_gate_logits') or model.last_gate_logits is None:
        return 0.0
    
    gate_logits = model.last_gate_logits
    batch_size = gate_logits.size(0)
    num_experts = gate_logits.size(1)
    
    _, expert_indices = torch.max(gate_logits, dim=1)
    
    expert_counts = torch.bincount(expert_indices, minlength=num_experts).float()
    
    ideal_count = batch_size / num_experts
    
    loss = ((expert_counts - ideal_count) ** 2).sum() / batch_size
    
    return loss

def calc_importance_loss_with_entropy(model, target_cv=0.1):
    """Balance expert usage while penalizing uniform distributions"""
    if not hasattr(model, 'last_gate_probs') or model.last_gate_probs is None:
        return 0.0
    
    gate_probs = model.last_gate_probs
    batch_size, num_experts = gate_probs.size()
    
    importance = gate_probs.mean(dim=0)
    
    importance_mean = importance.mean()
    importance_std = importance.std()
    cv = importance_std / (importance_mean + 1e-8)
    
    importance_loss = (cv - target_cv) ** 2
    
    sample_entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1)
    max_entropy = torch.log(torch.tensor(num_experts, device=gate_probs.device))
    entropy_penalty = (sample_entropy / max_entropy).mean()
    
    return importance_loss + entropy_penalty

def calc_topk_load_balancing_loss(model, k=1):
    """Load balancing specifically for top-k routing"""
    if not hasattr(model, 'last_gate_probs') or model.last_gate_probs is None:
        return 0.0
    
    gate_probs = model.last_gate_probs
    batch_size, num_experts = gate_probs.size()
    
    topk_probs, topk_indices = torch.topk(gate_probs, k, dim=1)
    
    expert_mask = torch.zeros_like(gate_probs)
    expert_mask.scatter_(1, topk_indices, 1)
    
    expert_usage = expert_mask.sum(dim=0) / batch_size
    ideal_usage = k / num_experts
    
    loss = ((expert_usage - ideal_usage) ** 2).mean()
    
    return loss

def calc_mutual_information_loss(model, labels, num_classes=10):
    """Maximize mutual information between expert selection and class labels"""
    if not hasattr(model, 'last_gate_logits') or model.last_gate_logits is None:
        return 0.0
    
    gate_logits = model.last_gate_logits
    batch_size, num_experts = gate_logits.size()
    
    _, expert_indices = torch.max(gate_logits, dim=1)
    
    joint_prob = torch.zeros(num_experts, num_classes, device=gate_logits.device)
    
    for e in range(num_experts):
        for c in range(num_classes):
            mask = (expert_indices == e) & (labels == c)
            joint_prob[e, c] = mask.float().sum() / batch_size
    
    p_expert = joint_prob.sum(dim=1, keepdim=True)
    p_class = joint_prob.sum(dim=0, keepdim=True)
    
    p_marginal_product = p_expert * p_class
    epsilon = 1e-10
    mi_terms = joint_prob * torch.log((joint_prob + epsilon) / (p_marginal_product + epsilon))
    mutual_info = mi_terms.sum()
    
    return -mutual_info
    