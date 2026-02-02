# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Heuristic-based zero-cost proxies for NAS.
- heuristic3: BatchNorm max outputs + Conv gradient magnitudes
- heuristic4: BatchNorm variance + first Conv weight variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import measure


@measure('heuristic3', bn=True, copy_net=True)
def compute_heuristic3(net, inputs, targets, loss_fn=F.cross_entropy, split_data=1):
    """
    Heuristic 3: Sum of BatchNorm max outputs + Mean of Conv gradient magnitudes
    """
    device = inputs.device
    net = net.to(device)
    net.train()
    
    bn_max_outputs = []
    conv_gradient_magnitudes = []
    
    def bn_max_hook(module, input, output):
        if isinstance(module, nn.BatchNorm2d):
            max_output = output.max().item()
            bn_max_outputs.append(max_output)
    
    def conv_gradient_hook(module, grad_input, grad_output):
        if isinstance(module, nn.Conv2d):
            if grad_output[0] is not None:
                grad_magnitude = grad_output[0].abs().mean().item()
                conv_gradient_magnitudes.append(grad_magnitude)
    
    hooks = []
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            hooks.append(layer.register_forward_hook(bn_max_hook))
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_full_backward_hook(conv_gradient_hook))
    
    # Forward and backward pass
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        outputs = net(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate final score
    sum_bn_max_output_score = sum(bn_max_outputs) if bn_max_outputs else 0.0
    mean_conv_gradient_score = (sum(conv_gradient_magnitudes) / len(conv_gradient_magnitudes)) if conv_gradient_magnitudes else 0.0
    
    score = sum_bn_max_output_score + mean_conv_gradient_score
    
    # Return as a list to match the expected format (will be summed in find_measures)
    return [torch.tensor([score])]


@measure('heuristic4', bn=True, copy_net=True)
def compute_heuristic4(net, inputs, targets, loss_fn=F.cross_entropy, split_data=1):
    """
    Heuristic 4: Sum of BatchNorm variance + First Conv weight variance
    """
    device = inputs.device
    net = net.to(device)
    net.train()
    
    bn_variances = []
    first_conv_weight_variance = None
    
    def bn_hook(module, input, output):
        if isinstance(module, nn.BatchNorm2d):
            variance = torch.var(output).item()
            bn_variances.append(variance)
    
    # Find first conv weight variance
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) and first_conv_weight_variance is None:
            first_conv_weight_variance = torch.var(layer.weight).item()
            break
    
    hooks = []
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            hooks.append(layer.register_forward_hook(bn_hook))
    
    # Forward and backward pass
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        
        outputs = net(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate final score
    sum_bn_variance_score = sum(bn_variances) if bn_variances else 0.0
    score = sum_bn_variance_score + (first_conv_weight_variance if first_conv_weight_variance is not None else 0.0)
    
    # Return as a list to match the expected format (will be summed in find_measures)
    return [torch.tensor([score])]
