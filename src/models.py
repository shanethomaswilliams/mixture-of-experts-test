import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.nn.utils as nn_utils
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import math
from scipy.stats import norm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetStyleExpert(nn.Module):
    """Expert using ResNet-style architecture with increasing widths"""
    def __init__(self, base_width=64, num_layers=4, num_blocks_per_layer=[2,2,2,2], num_classes=10):
        super().__init__()
        self.in_planes = base_width
        
        # Initial convolution to transform input to base width
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        
        # Create layers with increasing width (64, 128, 256, 512 for base_width=64)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            planes = base_width * (2 ** i)  # Width doubles at each layer
            stride = 1 if i == 0 else 2  # First layer has stride=1, others have stride=2
            self.layers.append(
                self._make_layer(BasicBlock, planes, num_blocks_per_layer[i], stride)
            )
        
        # Final classification layer
        final_planes = base_width * (2 ** (num_layers - 1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_planes, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        for layer in self.layers:
            out = layer(out)
        
        features = self.avgpool(out)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        
        return logits, features

class GatingNetwork(nn.Module):
    """Gate network that decides which expert to use"""
    def __init__(self, num_experts=4, input_channels=3, input_size=32, noise_std=1.0):
        super().__init__()
        self.noise_std = noise_std
        
        # Simple CNN for gate decision
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Gate decision layers
        self.gate_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
        
        # Initialize to uniform routing
        nn.init.zeros_(self.gate_fc[-1].weight)
        nn.init.constant_(self.gate_fc[-1].bias, 0.0)
        
    def forward(self, x):
        # Process image to make routing decision
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        
        # Get gate logits
        gate_logits = self.gate_fc(out)
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
            
        return gate_logits

class AdaptiveMoEWithSkip(nn.Module):
    """MoE with ResNet-style experts that adapt to match ResNet18 parameter count"""
    def __init__(self, num_experts=4, num_classes=10, use_skip=True, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.use_skip = use_skip
        self.top_k = top_k
        
        # ResNet18 target parameter count
        resnet18_params = 11173962
        
        # Define possible configurations (num_layers, blocks_per_layer)
        layer_configs = [
            (4, [2, 2, 2, 2]),  # Full ResNet18 structure
            (3, [2, 2, 2]),     # 3 layers
            (2, [2, 2]),        # 2 layers
            (1, [2]),           # Minimum: 1 layer
        ]
        
        best_config = None
        best_width = None
        best_diff = float('inf')
        
        for num_layers, num_blocks in layer_configs:
            low, high = 32, 512
            
            while low <= high:
                mid = (low + high) // 2
                
                temp_model = self._create_temp_model(mid, num_layers, num_blocks, num_experts)
                current_params = count_parameters(temp_model)
                
                diff = abs(current_params - resnet18_params)
                if diff < best_diff:
                    best_diff = diff
                    best_width = mid
                    best_config = (num_layers, num_blocks)
                
                if current_params < resnet18_params:
                    low = mid + 1
                else:
                    high = mid - 1
            
            if best_diff < resnet18_params * 0.01:
                break
        
        self.base_width = best_width
        self.num_layers, self.num_blocks = best_config
        
        print(f"Selected config for {num_experts} experts: width={best_width}, layers={self.num_layers}, blocks={self.num_blocks}")
        
        # Create the actual model with the best configuration
        self.experts = nn.ModuleList([
            ResNetStyleExpert(
                base_width=best_width, 
                num_layers=self.num_layers, 
                num_blocks_per_layer=self.num_blocks, 
                num_classes=num_classes
            ) 
            for _ in range(num_experts)
        ])
        
        # Gating network with noise support
        self.gate = GatingNetwork(num_experts=num_experts, noise_std=1.0)
        
        # Skip connection path
        if use_skip:
            # Skip path final width matches the expert's final layer width
            final_expert_width = best_width * (2 ** (self.num_layers - 1))
            
            self.skip_path = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, final_expert_width)
            )
            
            self.final_fc = nn.Linear(final_expert_width * 2, num_classes)
        
        # For load balancing loss
        self.last_gate_logits = None
        self.last_gate_probs = None
    
    def _create_temp_model(self, width, num_layers, num_blocks, num_experts):
        """Helper to create temporary model for parameter counting"""
        temp_experts = nn.ModuleList([
            ResNetStyleExpert(base_width=width, num_layers=num_layers, num_blocks_per_layer=num_blocks) 
            for _ in range(num_experts)
        ])
        temp_gate = GatingNetwork(num_experts=num_experts)
        
        final_expert_width = width * (2 ** (num_layers - 1))
        temp_skip = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, final_expert_width)
        )
        temp_final = nn.Linear(final_expert_width * 2, 10)
        
        return nn.ModuleList([temp_experts, temp_gate, temp_skip, temp_final])
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Get gate decisions
        gate_logits = self.gate(x)
        self.last_gate_logits = gate_logits
        
        # Apply softmax to get probabilities
        gate_probs = F.softmax(gate_logits, dim=1)
        self.last_gate_probs = gate_probs
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
        
        # Process through experts
        all_logits = []
        all_features = []
        
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i).any(dim=1)
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_logits, expert_features = self.experts[i](expert_input)
                all_logits.append((expert_logits, expert_mask, i))
                all_features.append((expert_features, expert_mask, i))
        
        # Combine expert outputs
        feature_dim = all_features[0][0].size(1) if all_features else self.base_width * (2 ** (self.num_layers - 1))
        final_logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        final_features = torch.zeros(batch_size, feature_dim, device=x.device)
        
        for logits, mask, expert_idx in all_logits:
            expert_weights = torch.zeros(batch_size, device=x.device)
            for k in range(self.top_k):
                expert_selected = (top_k_indices[:, k] == expert_idx)
                expert_weights[expert_selected] = top_k_probs[expert_selected, k]
            
            final_logits[mask] += logits * expert_weights[mask].unsqueeze(1)
        
        for features, mask, expert_idx in all_features:
            expert_weights = torch.zeros(batch_size, device=x.device)
            for k in range(self.top_k):
                expert_selected = (top_k_indices[:, k] == expert_idx)
                expert_weights[expert_selected] = top_k_probs[expert_selected, k]
            
            final_features[mask] += features * expert_weights[mask].unsqueeze(1)
        
        # Apply skip connection if enabled
        if self.use_skip:
            skip_features = self.skip_path(x)
            combined_features = torch.cat([final_features, skip_features], dim=1)
            final_output = self.final_fc(combined_features)
        else:
            final_output = final_logits
        
        return final_output

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

class SimpleExpert(nn.Module):
    """A simple expert using a single ResNet block"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution to transform input to expected dimensions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Single ResNet block
        self.block = BasicBlock(64, 64)
        
        # Downsample to get to classification
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Initial conv layer
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet block
        out = self.block(out)
        out = self.block(out)
        
        # Get features before final FC layer
        features = self.avgpool(out)
        features = features.view(features.size(0), -1)
        
        # Get final logits
        logits = self.fc(features)
        
        # Return both logits and features
        return logits, features

class TraditionalMoEWithSkip(nn.Module):
    """Traditional MoE with simple ResNet block experts and skip connections"""
    def __init__(self, num_experts=4, num_classes=10, use_skip=True, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.use_skip = use_skip
        self.top_k = top_k
        self.noise_std = 0.05
        
        # Create multiple simple experts
        self.experts = nn.ModuleList([SimpleExpert(num_classes=num_classes) for _ in range(num_experts)])
        
        # Gating network
        self.gate = GatingNetwork(num_experts=num_experts)
        
        # Skip connection path (lightweight feature extractor)
        if use_skip:
            self.skip_path = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 64)  # Changed to 64 to match expert features
            )
            
            # Combine features and make final prediction
            self.final_fc = nn.Linear(64 + 64, num_classes)  # Skip features + expert features
        
        # For load balancing loss
        self.last_gate_logits = None
        self.last_gate_probs = None
        
    def forward(self, x, training=True):
        batch_size = x.size(0)
        
        # Get gate decisions
        gate_logits = self.gate(x)

        if training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
            
        self.last_gate_logits = gate_logits  # Save for load balancing
        
        # Apply softmax to get probabilities
        gate_probs = F.softmax(gate_logits, dim=1)
        self.last_gate_probs = gate_probs
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
        
        # Process through experts
        all_logits = []
        all_features = []
        
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i).any(dim=1)
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_logits, expert_features = self.experts[i](expert_input)
                all_logits.append((expert_logits, expert_mask, i))
                all_features.append((expert_features, expert_mask, i))
        
        # Combine expert outputs
        final_logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        final_features = torch.zeros(batch_size, 64, device=x.device)  # Changed to 64
        
        for logits, mask, expert_idx in all_logits:
            # Find weights for this expert
            expert_weights = torch.zeros(batch_size, device=x.device)
            for k in range(self.top_k):
                expert_selected = (top_k_indices[:, k] == expert_idx)
                expert_weights[expert_selected] = top_k_probs[expert_selected, k]
            
            # Apply weighted logits
            final_logits[mask] += logits * expert_weights[mask].unsqueeze(1)
        
        for features, mask, expert_idx in all_features:
            # Find weights for this expert
            expert_weights = torch.zeros(batch_size, device=x.device)
            for k in range(self.top_k):
                expert_selected = (top_k_indices[:, k] == expert_idx)
                expert_weights[expert_selected] = top_k_probs[expert_selected, k]
            
            # Apply weighted features
            final_features[mask] += features * expert_weights[mask].unsqueeze(1)
        
        # Apply skip connection if enabled
        if self.use_skip:
            skip_features = self.skip_path(x)
            combined_features = torch.cat([final_features, skip_features], dim=1)
            final_output = self.final_fc(combined_features)
        else:
            final_output = final_logits
        
        return final_output
