import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import math

# 自定义异常类，用于在捕获到目标层的输出后停止前向传播
class StopForwardException(Exception):
    """用于停止前向传播的自定义异常"""
    def __init__(self, output):
        self.output = output  # 存储目标层的输出
        super().__init__("Forward propagation stopped at target layer")

class LLFCAnalyzer:
    def __init__(self, device=None, act=F.gelu):
        """
        初始化 LLFCAnalyzer 实例。

        Args:
            device (str, optional): 计算设备，如 'cuda:0' 或 'cpu'。默认 'cuda:0'。
            act (function, optional): 层中使用的激活函数，默认为 F.gelu。
        """
        self.activation_outputs = {}   # 存储捕获的层输出
        self.handles = []              # 存储钩子句柄
        self.layer_input = None        # 存储目标层的输入张量
        self.layer_output = None       # 存储目标层的输出张量
        self.act = act                 # 激活函数
        self.device = device if device is not None else 'cuda:0'
        self._module_cache = {}        # 用于缓存模块查找，加速后续调用

    def _get_target_module(self, model: nn.Module, target_layer: str) -> nn.Module:
        """
        内部辅助函数：查找并缓存模型中指定名称的模块。
        """
        cache_key = (id(model), target_layer)
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        for name, module in model.named_modules():
            if name == target_layer:
                self._module_cache[cache_key] = module
                return module
        raise ValueError(f"Target layer '{target_layer}' not found in model")

    def clear_hooks(self) -> None:
        """
        移除所有已注册的钩子，并清空存储的激活输出。
        """
        if self.handles:
            for handle in self.handles:
                handle.remove()
        self.handles = []
        self.activation_outputs.clear()

    def register_hooks(self, model: nn.Module):
        """
        为模型中所有 nn.Linear 层注册前向钩子以捕获输出。
        """
        self.clear_hooks()
        def get_activation(name):
            def hook(module, input, output):
                self.activation_outputs[name] = output
            return hook
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_activation(name))
                self.handles.append(handle)

    def register_single_hook(self, model: nn.Module, target_layer: str) -> bool:
        """
        为模型的指定层注册钩子，捕获该层输出。
        """
        self.clear_hooks()
        target_module = self._get_target_module(model, target_layer)
        handle = target_module.register_forward_hook(lambda module, inp, out: self.activation_outputs.update({target_layer: out}))
        self.handles.append(handle)
        return True

    def register_single_hook_and_stop(self, model: nn.Module, target_layer: str) -> bool:
        """
        为模型的指定层注册钩子，捕获输出后立即停止前向传播。
        """
        self.clear_hooks()
        target_module = self._get_target_module(model, target_layer)
        def hook(module, inp, out):
            self.activation_outputs[target_layer] = out
            raise StopForwardException(out)
        handle = target_module.register_forward_hook(hook)
        self.handles.append(handle)
        return True

    def register_input_hook(self, model: nn.Module, target_layer: str) -> bool:
        """
        注册钩子捕获指定层的输入和输出。
        """
        self.clear_hooks()
        target_module = self._get_target_module(model, target_layer)
        def hook(module, inp, out):
            # 捕获第一个输入元素，并转移到目标设备
            self.layer_input = inp[0].detach().to(self.device)
            self.layer_output = out.detach().to(self.device)
        handle = target_module.register_forward_hook(hook)
        self.handles.append(handle)
        return True

    def get_layer_input(self, model: nn.Module, target_layer: str, model_input: torch.Tensor) -> torch.Tensor:
        """
        获取模型在指定层的输入张量。
        """
        self.register_input_hook(model, target_layer)
        model.eval()
        with torch.no_grad():
            _ = model(model_input)
        layer_input = self.layer_input
        self.clear_hooks()
        return layer_input

    def feed_layer(self, model: nn.Module, target_layer: str, layer_input: torch.Tensor) -> torch.Tensor:
        """
        将预先捕获的层输入传递给指定层，并返回该层的输出。
        """
        target_module = self._get_target_module(model, target_layer)
        # 如果目标模块不在目标设备上，则移动；避免重复转换
        if next(target_module.parameters(), None) is not None:
            if next(target_module.parameters()).device != self.device:
                target_module = target_module.to(self.device)
        target_module.eval()
        with torch.no_grad():
            output = target_module(layer_input)
        return output

    def apply_activation(self, x: torch.Tensor, activation_type: str, params: Dict) -> torch.Tensor:
        """应用激活函数"""
        if activation_type == 'relu':
            return F.relu(x, inplace=params.get('inplace', False))
        elif activation_type == 'gelu':
            return F.gelu(x, approximate="tanh")
        return x

    @torch.no_grad()
    def compute_llfc_loss(self, layer_outputs: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        """
        计算多个模型在指定层的 LLFC 损失，衡量它们输出的一致性。
        """
        # 利用堆叠和广播进行向量化计算
        stacked = torch.stack(layer_outputs, dim=0)  # (num_models, batch_size, seq_length, hidden_dim)
        batch_size = stacked.size(1)
        weights_tensor = torch.tensor(weights, device=stacked.device).view(-1, 1, 1, 1)
        weighted_sum = (stacked * weights_tensor).sum(dim=0)
        activated_weighted_sum = self.act(weighted_sum)
        activated_outputs = self.act(stacked)
        activated_sum = (activated_outputs * weights_tensor).sum(dim=0)
        difference = activated_sum - activated_weighted_sum
        sample_losses = torch.norm(difference.view(batch_size, -1), p=2, dim=1)
        return sample_losses

    @torch.no_grad()
    def compute_llfc_loss_by_row(self, sample_input, models: List[nn.Module], weights: List[float],
                                 layer_name: str, activation_type='gelu'):
        """
        计算多个模型在指定层的理论 LLFC 损失，按每个隐藏维度计算一致性损失。
        差异定义为 sigma(avg(Wx)) - avg(sigma(Wx))，返回形状为 (hidden_dim,) 的张量，
        每个元素代表该隐藏维度上的损失。

        注意：不改变输入和输出的数据格式。
        """
        # 使用第一个模型作为基准，获取目标层输入
        base_model = models[0]
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)
        # 其他模型的输出
        layer_outputs = [self.feed_layer(model, layer_name, layer_input) for model in models[1:]]
        if len(weights) != len(layer_outputs):
            raise ValueError("The number of weights must match the number of layer outputs.")
        # 激活参数设置
        activation_params = {'approximate': 'none'} if activation_type == 'gelu' else {}
        # 堆叠输出，并展平 (batch_size * patch_size, hidden_dim)
        stacked_outputs = torch.stack(layer_outputs, dim=0)  # (num_models, batch_size, patch_size, hidden_dim)
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
        """
        计算多个模型在指定层的规范化 LLFC 损失，使损失对层输出缩放不敏感。
        """
        stacked = torch.stack(layer_outputs, dim=0)  # (num_models, batch_size, seq_length, hidden_dim)
        batch_size = stacked.size(1)
        weights_tensor = torch.tensor(weights, device=stacked.device).view(-1, 1, 1, 1)
        weighted_sum = (stacked * weights_tensor).sum(dim=0)
        activated_weighted_sum = self.act(weighted_sum)
        activated_outputs = self.act(stacked)
        activated_sum = (activated_outputs * weights_tensor).sum(dim=0)
        average_activated_output = activated_sum
        difference = average_activated_output - activated_weighted_sum
        norm_a = torch.norm(activated_weighted_sum.view(batch_size, -1), p=2, dim=1, keepdim=True) + 1e-8
        normalized_difference = difference / norm_a.unsqueeze(-1)
        sample_losses = torch.norm(normalized_difference.view(batch_size, -1), p=2, dim=1)
        return sample_losses

    @torch.no_grad()
    def compute_llfc_loss_without_merge(self, layer_outputs: List[torch.Tensor], weights: List[float], base_model_output) -> torch.Tensor:
        """
        计算多个模型在指定层的 LLFC 损失，不使用基准模型的层输入进行比较。
        """
        stacked = torch.stack(layer_outputs, dim=0)  # (num_models, batch_size, seq_length, hidden_dim)
        batch_size = stacked.size(1)
        weights_tensor = torch.tensor(weights, device=stacked.device).view(-1, 1, 1, 1)
        activated_weighted_sum = self.act(base_model_output)
        activated_outputs = self.act(stacked)
        activated_sum = (activated_outputs * weights_tensor).sum(dim=0)
        average_activated_output = activated_sum
        difference = average_activated_output - activated_weighted_sum
        sample_losses = torch.norm(difference.view(batch_size, -1), p=2, dim=1)
        return sample_losses

    def analyze_layer(self, layer_name: str, models: List[nn.Module], weights: List[float],
                      sample_input: torch.Tensor) -> torch.Tensor:
        """
        分析指定层的 LLFC 损失（新版方法）。

        说明：
            - 使用第一个模型作为基准获取目标层输入，
              其他模型利用相同输入获得输出后计算 LLFC 损失。
        """
        try:
            base_model = models[0]
            layer_input = self.get_layer_input(base_model, layer_name, sample_input)
            layer_outputs = [self.feed_layer(model, layer_name, layer_input) for model in models[1:]]
            batch_losses = self.compute_llfc_loss(layer_outputs, weights[1:])
            return batch_losses  # 返回形状为 (batch_size,) 的张量
        except Exception as e:
            print(f"Error in analyze_layer: {e}")
            raise

    def analyze_layer_without_merge(self, layer_name: str, models: List[nn.Module], weights: List[float],
                                      sample_input: torch.Tensor, base_model: nn.Module) -> torch.Tensor:
        """
        分析指定层的 LLFC 损失，不使用基准模型的层输入进行比较。
        """
        try:
            layer_outputs = [self.get_intermediate_output(model, layer_name, sample_input) for model in models]
            base_output = self.get_intermediate_output(base_model, layer_name, sample_input)
            batch_losses = self.compute_llfc_loss_without_merge(layer_outputs, weights, base_output)
            return batch_losses
        except Exception as e:
            print(f"Error in analyze_layer_without_merge: {e}")
            raise

    def analyze_layer_output(self, layer_name: str, models: List[nn.Module], sample_input: torch.Tensor) -> List[torch.Tensor]:
        """
        返回基准模型以外的模型在指定层的输出列表。
        """
        base_model = models[0]
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)
        layer_outputs = [self.feed_layer(model, layer_name, layer_input) for model in models[1:]]
        return layer_outputs

    def get_intermediate_output(self, model: nn.Module, target_layer: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        获取模型在指定层的输出，并在该层后停止前向传播。
        """
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
