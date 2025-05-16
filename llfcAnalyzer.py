import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class StopForwardException(Exception):
    def __init__(self, output):
        self.output = output
        super().__init__("Forward propagation stopped at target layer")

class LLFCAnalyzer:
    def __init__(self, device=None, act=F.gelu):
        self.activation_outputs = {}
        self.handles = []
        self.layer_input = None
        self.layer_output = None
        self.act = act
        self.device = device if device is not None else 'cuda:0'
        self._module_cache = {}
        self._weight_cache = {}

    def _get_target_module(self, model: nn.Module, target_layer: str) -> nn.Module:
        cache_key = (id(model), target_layer)
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        for name, module in model.named_modules():
            if name == target_layer:
                self._module_cache[cache_key] = module
                return module
        raise ValueError(f"Target layer '{target_layer}' not found in model")

    def _get_weights_tensor(self, weights: List[float], num_models: int, device: torch.device) -> torch.Tensor:
        key = (tuple(weights), num_models, device)
        if key in self._weight_cache:
            return self._weight_cache[key]
        weight_tensor = torch.tensor(weights, device=device).view(num_models, 1, 1)
        self._weight_cache[key] = weight_tensor
        return weight_tensor

    def clear_hooks(self) -> None:
        if self.handles:
            for handle in self.handles:
                handle.remove()
        self.handles = []
        self.activation_outputs.clear()

    def register_single_hook_and_stop(self, model: nn.Module, target_layer: str) -> bool:
        self.clear_hooks()
        target_module = self._get_target_module(model, target_layer)
        def hook(module, inp, out):
            self.activation_outputs[target_layer] = out
            raise StopForwardException(out)
        handle = target_module.register_forward_hook(hook)
        self.handles.append(handle)
        return True

    def register_input_hook(self, model: nn.Module, target_layer: str) -> bool:
        self.clear_hooks()
        target_module = self._get_target_module(model, target_layer)
        def hook(module, inp, out):
            self.layer_input = inp[0].detach().to(self.device)
            self.layer_output = out.detach().to(self.device)
        handle = target_module.register_forward_hook(hook)
        self.handles.append(handle)
        return True

    def get_layer_input(self, model: nn.Module, target_layer: str, model_input: torch.Tensor) -> torch.Tensor:
        self.register_input_hook(model, target_layer)
        model.eval()
        with torch.no_grad():
            _ = model(model_input)
        layer_input = self.layer_input
        self.clear_hooks()
        return layer_input

    def feed_layer(self, model: nn.Module, target_layer: str, layer_input: torch.Tensor) -> torch.Tensor:
        target_module = self._get_target_module(model, target_layer)
        if next(target_module.parameters(), None) is not None:
            if next(target_module.parameters()).device != self.device:
                target_module = target_module.to(self.device)
        target_module.eval()
        with torch.no_grad():
            output = target_module(layer_input)
        return output

    def apply_activation(self, x: torch.Tensor, activation_type: str, params: Dict) -> torch.Tensor:
        if activation_type == 'relu':
            return torch.nn.functional.relu(x, inplace=params.get('inplace', False))
        elif activation_type == 'gelu':
            return torch.nn.functional.gelu(x, approximate="tanh")
        return x

    @torch.no_grad()
    def compute_llfc_loss_by_row(self, sample_input, models: List[nn.Module], weights: List[float],
                                 layer_name: str, activation_type='gelu'):
        base_model = models[0]
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)
        layer_outputs = [self.feed_layer(model, layer_name, layer_input) for model in models[1:]]
        if len(weights) != len(layer_outputs):
            raise ValueError("The number of weights must match the number of layer outputs.")
        activation_params = {'approximate': 'none'} if activation_type == 'gelu' else {}
        stacked_outputs = torch.stack(layer_outputs, dim=0)
        num_models, batch_size, patch_size, hidden_dim = stacked_outputs.shape
        flattened_outputs = stacked_outputs.view(num_models, -1, hidden_dim)
        weights_tensor = self._get_weights_tensor(weights, num_models, flattened_outputs.device)
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

    def analyze_layer(self, layer_name: str, models: List[nn.Module], weights: List[float],
                      sample_input: torch.Tensor) -> torch.Tensor:
        try:
            base_model = models[0]
            layer_input = self.get_layer_input(base_model, layer_name, sample_input)
            layer_outputs = [self.feed_layer(model, layer_name, layer_input) for model in models[1:]]
            batch_losses = self.compute_llfc_loss(layer_outputs, weights[1:])
            return batch_losses
        except Exception as e:
            print(f"Error in analyze_layer: {e}")
            raise

    def analyze_layer_without_merge(self, layer_name: str, models: List[nn.Module], weights: List[float],
                                   sample_input: torch.Tensor, base_model: nn.Module) -> torch.Tensor:
        try:
            layer_outputs = [self.get_intermediate_output(model, layer_name, sample_input) for model in models]
            base_output = self.get_intermediate_output(base_model, layer_name, sample_input)
            batch_losses = self.compute_llfc_loss_without_merge(layer_outputs, weights, base_output)
            return batch_losses
        except Exception as e:
            print(f"Error in analyze_layer_without_merge: {e}")
            raise

    def analyze_layer_output(self, layer_name: str, models: List[nn.Module], sample_input: torch.Tensor) -> List[torch.Tensor]:
        base_model = models[0]
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)
        layer_outputs = [self.feed_layer(model, layer_name, layer_input) for model in models[1:]]
        return layer_outputs

    def get_intermediate_output(self, model: nn.Module, target_layer: str, input_tensor: torch.Tensor) -> torch.Tensor:
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
