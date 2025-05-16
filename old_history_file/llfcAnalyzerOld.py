import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import math

class StopForwardException(Exception):
    """Custom exception to stop forward propagation at a target layer."""
    def __init__(self, output):
        self.output = output
        super().__init__("Forward propagation stopped at target layer")

class LLFCAnalyzer:
    def __init__(self, device=None, act=F.gelu):
        self.activation_outputs = {}
        self.handles = None
        self.layer_input = None
        self.layer_output = None
        self.act = act
        self.device = device if device is not None else 'cuda:0'

    def register_hooks(self, model: nn.Module):
        if self.handles is not None:
            for handle in self.handles:
                handle.remove()
        self.handles = []
        self.activation_outputs.clear()

        def get_activation(name):
            def hook(module, input, output):
                self.activation_outputs[name] = output
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_activation(name))
                self.handles.append(handle)

    def register_single_hook(self, model: nn.Module, target_layer: str) -> bool:
        self.clear_hooks()

        def get_activation(name):
            def hook(module, input, output):
                self.activation_outputs[name] = output
            return hook

        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module
                break
        if target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")
        handle = target_module.register_forward_hook(get_activation(target_layer))
        self.handles.append(handle)
        return True

    def register_single_hook_and_stop(self, model: nn.Module, target_layer: str) -> bool:
        self.clear_hooks()

        def get_activation_and_stop(name):
            def hook(module, input, output):
                self.activation_outputs[name] = output
                raise StopForwardException(output)
            return hook

        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module
                break
        if target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")
        handle = target_module.register_forward_hook(get_activation_and_stop(target_layer))
        self.handles.append(handle)
        return True

    def register_input_hook(self, model: nn.Module, target_layer: str) -> bool:
        self.clear_hooks()

        def hook(module, input, output):
            self.layer_input = input[0].detach().to(self.device)
            self.layer_output = output.detach().to(self.device)

        for name, module in model.named_modules():
            if name == target_layer:
                handle = module.register_forward_hook(hook)
                self.handles.append(handle)
                return True
        raise ValueError(f"Target layer '{target_layer}' not found in model")

    def get_layer_input(self, model: nn.Module, target_layer: str, model_input: torch.Tensor) -> torch.Tensor:
        self.register_input_hook(model, target_layer)
        model.eval()
        with torch.no_grad():
            _ = model(model_input)
        layer_input = self.layer_input
        self.clear_hooks()
        return layer_input

    def feed_layer(self, model: nn.Module, target_layer: str, layer_input: torch.Tensor) -> torch.Tensor:
        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module.to(self.device)
                break
        if target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")
        target_module.eval()
        with torch.no_grad():
            output = target_module(layer_input)
        return output

    def clear_hooks(self) -> None:
        if hasattr(self, 'handles') and self.handles:
            for handle in self.handles:
                handle.remove()
        self.handles = []
        self.activation_outputs.clear()

    @torch.no_grad()
    def compute_llfc_loss(self, layer_outputs: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        num_models = len(layer_outputs)
        batch_size = layer_outputs[0].size(0)
        seq_length, hidden_dim = layer_outputs[0].size(1), layer_outputs[0].size(2)

        total_weight = sum(weights)
        tolerance = 1e-4
        if not math.isclose(total_weight, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Weights must sum to 1 within a tolerance of {tolerance}. "
                f"Current sum is {total_weight}."
            )

        weighted_sum = torch.zeros_like(layer_outputs[0])
        for output, weight in zip(layer_outputs, weights):
            weighted_sum += output * weight

        activated_weighted_sum = self.act(weighted_sum)

        activated_sum = torch.zeros_like(layer_outputs[0])
        for output, weight in zip(layer_outputs, weights):
            activated_output = self.act(output)
            activated_sum += activated_output * weight

        average_activated_output = activated_sum
        difference = average_activated_output - activated_weighted_sum
        sample_losses = torch.norm(difference.view(batch_size, -1), p=2, dim=1)
        return sample_losses

    def apply_activation(self, x: torch.Tensor, activation_type: str, params: Dict) -> torch.Tensor:
        if activation_type == 'relu':
            return torch.nn.functional.relu(x, inplace=params.get('inplace', False))
        elif activation_type == 'gelu':
            return torch.nn.functional.gelu(x, approximate="tanh")
        return x

    @torch.no_grad()
    def compute_llfc_loss_by_row(
        self,
        sample_input,
        models,
        weights: List[float],
        layer_name: str,
        activation_type='gelu'
    ):
        base_model = models[0]
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)

        layer_outputs = []
        for model in models[1:]:
            output = self.feed_layer(model, layer_name, layer_input)
            layer_outputs.append(output)
        if len(weights) != len(layer_outputs):
            raise ValueError("The number of weights must match the number of layer outputs.")

        num_models = len(layer_outputs)
        if activation_type == 'gelu':
            activation_params = {'approximate': 'none'}
        if activation_type == 'none':
            hidden_dim = layer_outputs[0].shape[-1]
            return [1.0] * hidden_dim

        stacked_outputs = torch.stack(layer_outputs, dim=0)
        num_models, batch_size, patch_size, hidden_dim = stacked_outputs.shape
        flattened_outputs = stacked_outputs.view(num_models, -1, hidden_dim)
        weights_tensor = torch.tensor(weights, device=flattened_outputs.device).view(num_models, 1, 1)
        weighted_avg_Wx = (flattened_outputs * weights_tensor).sum(dim=0)
        a = self.apply_activation(weighted_avg_Wx, activation_type, activation_params)
        activated_Wx = self.apply_activation(flattened_outputs, activation_type, activation_params)
        weighted_avg_activated_Wx = (activated_Wx * weights_tensor).sum(dim=0)
        diff = a - weighted_avg_activated_Wx
        if activation_type == 'relu':
            mask = (a > 0) | (weighted_avg_activated_Wx > 0)
            diff = diff * mask
        squared_diff = diff ** 2
        dim_loss = squared_diff.mean(dim=0)
        return dim_loss

    @torch.no_grad()
    def compute_norm_llfc_loss(self, layer_outputs: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        num_models = len(layer_outputs)
        batch_size = layer_outputs[0].size(0)
        seq_length, hidden_dim = layer_outputs[0].size(1), layer_outputs[0].size(2)

        total_weight = sum(weights)
        tolerance = 1e-4
        if not math.isclose(total_weight, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Weights must sum to 1 within a tolerance of {tolerance}. "
                f"Current sum is {total_weight}."
            )

        weighted_sum = torch.zeros_like(layer_outputs[0])
        for output, weight in zip(layer_outputs, weights):
            weighted_sum += output * weight

        activated_weighted_sum = self.act(weighted_sum)
        activated_sum = torch.zeros_like(layer_outputs[0])
        for output, weight in zip(layer_outputs, weights):
            activated_output = self.act(output)
            activated_sum += activated_output * weight
        average_activated_output = activated_sum
        difference = average_activated_output - activated_weighted_sum

        norm_a = torch.norm(activated_weighted_sum.view(batch_size, -1), p=2, dim=1, keepdim=True)
        epsilon = 1e-8
        norm_a = norm_a + epsilon
        normalized_difference = difference / norm_a.unsqueeze(-1)
        sample_losses = torch.norm(normalized_difference.view(batch_size, -1), p=2, dim=1)
        return sample_losses

    @torch.no_grad()
    def compute_llfc_loss_without_merge(self, layer_outputs: List[torch.Tensor], weights: List[float], base_model_output) -> torch.Tensor:
        num_models = len(layer_outputs)
        batch_size = layer_outputs[0].size(0)
        seq_length, hidden_dim = layer_outputs[0].size(1), layer_outputs[0].size(2)

        total_weight = sum(weights)
        tolerance = 1e-4
        if not math.isclose(total_weight, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Weights must sum to 1 within a tolerance of {tolerance}. "
                f"Current sum is {total_weight}."
            )

        weighted_sum = base_model_output
        activated_weighted_sum = self.act(weighted_sum)
        activated_sum = torch.zeros_like(layer_outputs[0])
        for output, weight in zip(layer_outputs, weights):
            activated_output = self.act(output)
            activated_sum += activated_output * weight
        average_activated_output = activated_sum
        difference = average_activated_output - activated_weighted_sum
        sample_losses = torch.norm(difference.view(batch_size, -1), p=2, dim=1)
        return sample_losses

    def analyze_layer(
            self,
            layer_name: str,
            models: List[nn.Module],
            weights: List[float],
            sample_input: torch.Tensor,
    ) -> torch.Tensor:
        all_losses = []
        try:
            base_model = models[0]
            layer_input = self.get_layer_input(base_model, layer_name, sample_input)
            layer_outputs = []
            for model in models[1:]:
                output = self.feed_layer(model, layer_name, layer_input)
                layer_outputs.append(output)
            batch_losses = self.compute_llfc_loss(layer_outputs, weights[1:])
            all_losses.append(batch_losses)
            return torch.cat(all_losses)
        except Exception as e:
            print(f"Error in analyze_layer: {e}")
            raise

    def analyze_layer_without_merge(
            self,
            layer_name: str,
            models: List[nn.Module],
            weights: List[float],
            sample_input: torch.Tensor,
            base_model: nn.Module
    ) -> torch.Tensor:
        all_losses = []
        try:
            layer_outputs = []
            for model in models:
                output = self.get_intermediate_output(model, layer_name, sample_input)
                layer_outputs.append(output)
            base_output = self.get_intermediate_output(base_model, layer_name, sample_input)
            batch_losses = self.compute_llfc_loss_without_merge(layer_outputs, weights, base_output)
            all_losses.append(batch_losses)
            return torch.cat(all_losses)
        except Exception as e:
            print(f"Error in analyze_layer: {e}")
            raise

    def analyze_layer_output(
            self,
            layer_name: str,
            models: List[nn.Module],
            sample_input: torch.Tensor,
    ) -> List[torch.Tensor]:
        base_model = models[0]
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)
        layer_outputs = []
        for model in models[1:]:
            output = self.feed_layer(model, layer_name, layer_input)
            layer_outputs.append(output)
        return layer_outputs

    def get_intermediate_output(
            self,
            model: nn.Module,
            target_layer: str,
            input_tensor: torch.Tensor
    ) -> torch.Tensor:
        self.register_single_hook_and_stop(model, target_layer)
        try:
            model.eval()
            with torch.no_grad():
                _ = model(input_tensor)
        except StopForwardException as e:
            return self.activation_outputs[target_layer]
        finally:
            self.clear_hooks()
        raise ValueError(f"Failed to get output for layer {target_layer}")
