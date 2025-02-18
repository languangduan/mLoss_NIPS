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
        super().__init__("Forward propagation stopped at target layer")  # 初始化父类异常信息


# LLFCAnalyzer 类用于分析多个模型在特定层的输出一致性
class LLFCAnalyzer:
    def __init__(self, device=None, act=F.gelu):
        """
        初始化 LLFCAnalyzer 实例。

        Args:
            device (str, optional): 计算设备，如 'cuda:0' 或 'cpu'。如果未提供，默认为 'cuda:2'。
            act (function, optional): 层中使用的激活函数，默认为 GELU 激活函数（F.gelu）。
        """
        # self.model = model  # (注释掉的代码，可能用于未来扩展)
        self.activation_outputs = {}  # 存储捕获的层输出，格式为 {layer_name: output_tensor}
        self.handles = None  # 存储钩子句柄，以便后续移除
        self.layer_input = None  # 存储目标层的输入张量
        self.layer_output = None  # 存储目标层的输出张量
        self.act = act  # 激活函数，用于处理层输出
        self.device = device if device is not None else 'cuda:0'  # 设置计算设备，默认使用 'cuda:2'
        # self.register_hooks()  # (注释掉的代码，可能用于未来扩展)

    def register_hooks(self, model: nn.Module):
        """
        为模型中所有指定类型的层（如 nn.Linear）注册前向钩子，以捕获这些层的输出。

        Args:
            model (nn.Module): 要注册钩子的模型。

        功能:
            1. 清除之前注册的所有钩子，防止钩子重复注册。
            2. 遍历模型的所有命名模块，找到指定类型的层（如 nn.Linear）。
            3. 为每个符合条件的层注册前向钩子，并将钩子句柄存储起来以便后续移除。
            4. 在前向传播过程中，钩子函数会将层的输出存储到 activation_outputs 字典中。
        """
        # 清除之前注册的钩子和存储的输出
        if self.handles is not None:
            for handle in self.handles:
                handle.remove()  # 移除每一个钩子
        self.handles = []  # 重置钩子句柄列表
        self.activation_outputs.clear()  # 清空存储的激活输出

        def get_activation(name):
            """
            定义钩子函数，用于捕获指定层的输出。

            Args:
                name (str): 层的名称。

            Returns:
                function: 钩子函数，接收 module, input, output 三个参数。
            """
            def hook(module, input, output):
                self.activation_outputs[name] = output  # 将输出存储到 activation_outputs 字典中
            return hook

        # 遍历模型的所有命名模块，找到需要监控的层类型并注册钩子
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):  # 检查模块类型是否为 nn.Linear（可根据需要调整）
                handle = module.register_forward_hook(get_activation(name))  # 注册前向钩子
                self.handles.append(handle)  # 保存钩子句柄

    def register_single_hook(
            self,
            model: nn.Module,
            target_layer: str
    ) -> bool:
        """
        为模型的指定单层注册钩子，以捕获该层的输出。

        Args:
            model (nn.Module): 要注册钩子的模型。
            target_layer (str): 目标层的名称。

        Returns:
            bool: 如果成功注册钩子，返回 True。

        功能:
            1. 清除现有的钩子和存储的输出，确保钩子不会重复注册。
            2. 遍历模型的所有命名模块，找到名称匹配的目标层。
            3. 为目标层注册前向钩子，并将钩子句柄存储起来。
            4. 在前向传播过程中，钩子函数会将目标层的输出存储到 activation_outputs 字典中。
        """
        self.clear_hooks()  # 清除现有的钩子和存储的输出

        def get_activation(name):
            """
            定义钩子函数，用于捕获指定层的输出。

            Args:
                name (str): 层的名称。

            Returns:
                function: 钩子函数，接收 module, input, output 三个参数。
            """
            def hook(module, input, output):
                self.activation_outputs[name] = output  # 将输出存储到 activation_outputs 字典中
            return hook

        # 遍历模型的所有命名模块，找到目标层
        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module  # 找到目标层
                break

        # 如果未找到目标层，抛出异常
        if target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")

        # 注册前向钩子并保存句柄
        handle = target_module.register_forward_hook(get_activation(target_layer))
        self.handles.append(handle)
        return True  # 返回成功注册的标志

    def register_single_hook_and_stop(
            self,
            model: nn.Module,
            target_layer: str
    ) -> bool:
        """
        为模型的指定单层注册前向钩子，并在该层后停止前向传播。

        Args:
            model (nn.Module): 要注册钩子的模型。
            target_layer (str): 目标层的名称。

        Returns:
            bool: 如果成功注册钩子，返回 True。

        功能:
            1. 清除现有的钩子和存储的输出，确保钩子不会重复注册。
            2. 遍历模型的所有命名模块，找到名称匹配的目标层。
            3. 为目标层注册前向钩子，钩子函数在捕获到输出后会引发 StopForwardException 异常，停止前向传播。
            4. 将钩子句柄存储起来以便后续移除。
        """
        self.clear_hooks()  # 清除现有的钩子和存储的输出

        def get_activation_and_stop(name):
            """
            定义钩子函数，用于捕获指定层的输出并停止前向传播。

            Args:
                name (str): 层的名称。

            Returns:
                function: 钩子函数，接收 module, input, output 三个参数。
            """
            def hook(module, input, output):
                self.activation_outputs[name] = output  # 将输出存储到 activation_outputs 字典中
                raise StopForwardException(output)  # 引发自定义异常，停止前向传播
            return hook

        # 遍历模型的所有命名模块，找到目标层
        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module  # 找到目标层
                break

        # 如果未找到目标层，抛出异常
        if target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")

        # 注册前向钩子并保存句柄
        handle = target_module.register_forward_hook(get_activation_and_stop(target_layer))
        self.handles.append(handle)
        return True  # 返回成功注册的标志

    def register_input_hook(self, model: nn.Module, target_layer: str) -> bool:
        """
        注册钩子来捕获指定层的输入和输出。

        Args:
            model (nn.Module): 要注册钩子的模型。
            target_layer (str): 目标层的名称。

        Returns:
            bool: 如果成功注册钩子，返回 True。

        功能:
            1. 清除现有的钩子和存储的输出，确保钩子不会重复注册。
            2. 定义钩子函数，用于捕获指定层的输入和输出，并将其存储到类的属性中。
            3. 遍历模型的所有命名模块，找到名称匹配的目标层。
            4. 为目标层注册前向钩子，并将钩子句柄存储起来。
            5. 如果未找到目标层，抛出异常。
        """
        self.clear_hooks()  # 清除现有的钩子和存储的输出

        def hook(module, input, output):
            """
            钩子函数，用于捕获指定层的输入和输出。

            Args:
                module (nn.Module): 当前模块。
                input (tuple): 模块的输入。
                output (torch.Tensor): 模块的输出。
            """
            # input 是一个 tuple，通常我们需要第一个元素作为实际输入
            self.layer_input = input[0].detach().to(self.device)  # 捕获并存储输入张量，转移到指定设备
            self.layer_output = output.detach().to(self.device)  # 捕获并存储输出张量，转移到指定设备

        # 遍历模型的所有命名模块，找到目标层
        for name, module in model.named_modules():
            if name == target_layer:
                handle = module.register_forward_hook(hook)  # 注册前向钩子
                self.handles.append(handle)  # 保存钩子句柄
                return True  # 返回成功注册的标志

        # 如果未找到目标层，抛出异常
        raise ValueError(f"Target layer '{target_layer}' not found in model")

    def get_layer_input(self, model: nn.Module, target_layer: str,
                        model_input: torch.Tensor) -> torch.Tensor:
        """
        获取模型中特定层的输入张量。

        Args:
            model (nn.Module): 要分析的模型。
            target_layer (str): 目标层的名称。
            model_input (torch.Tensor): 模型的输入张量，形状为 (batch_size, ...)。

        Returns:
            torch.Tensor: 目标层的输入张量，形状与目标层输入匹配。

        功能:
            1. 注册输入钩子，以捕获指定层的输入和输出。
            2. 将模型设置为评估模式，并执行前向传播以触发钩子。
            3. 从存储的 layer_input 属性中提取目标层的输入张量。
            4. 清除所有钩子，避免对后续操作产生影响。
        """
        self.register_input_hook(model, target_layer)  # 注册输入钩子以捕获指定层的输入和输出

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            _ = model(model_input)  # 执行前向传播，触发钩子

        layer_input = self.layer_input  # 获取捕获到的层输入
        self.clear_hooks()  # 移除钩子，避免影响后续操作

        return layer_input  # 返回目标层的输入张量

    def feed_layer(self, model: nn.Module, target_layer: str,
                   layer_input: torch.Tensor) -> torch.Tensor:
        """
        将预先捕获的层输入直接传递给指定层，并获取该层的输出。

        Args:
            model (nn.Module): 要分析的模型。
            target_layer (str): 目标层的名称。
            layer_input (torch.Tensor): 要传递给目标层的输入张量，形状与目标层输入匹配。

        Returns:
            torch.Tensor: 目标层的输出张量，形状与目标层输出匹配。

        功能:
            1. 遍历模型的所有命名模块，找到名称匹配的目标层。
            2. 将目标层移动到指定设备，并设置为评估模式。
            3. 禁用梯度计算，并将层输入传递给目标层，获取输出。
        """
        # 遍历模型的所有命名模块，找到目标层
        target_module = None
        for name, module in model.named_modules():
            if name == target_layer:
                target_module = module.to(self.device)  # 移动目标模块到指定设备
                break

        # 如果未找到目标层，抛出异常
        if target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")

        # 设置目标模块为评估模式，并禁用梯度计算
        target_module.eval()
        with torch.no_grad():
            output = target_module(layer_input)  # 将输入张量传递给目标层，获取输出

        return output  # 返回目标层的输出张量

    def clear_hooks(self) -> None:
        """
        移除所有已注册的钩子，并清空存储的激活输出。

        功能:
            1. 检查是否有已注册的钩子。
            2. 如果存在，逐个移除所有钩子。
            3. 重置钩子句柄列表和激活输出字典。
        """
        if hasattr(self, 'handles') and self.handles:
            for handle in self.handles:
                handle.remove()  # 移除每一个钩子
        self.handles = []  # 重置钩子句柄列表
        self.activation_outputs.clear()  # 清空激活输出字典




###仅适用于前几层都已经merge完了的情况
    @torch.no_grad()
    def compute_llfc_loss(self, layer_outputs: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        """
        计算多个模型在指定层的 LLFC 损失，衡量它们在该层输出的一致性。

        Args:
            layer_outputs (List[torch.Tensor]): 每个模型在目标层的输出张量列表。
                每个元素的形状为 (batch_size, seq_length, hidden_dim)。
            weights (List[float]): 每个模型在加权和中的权重列表。
                长度应与 layer_outputs 相同，且权重总和接近 1。

        Returns:
            torch.Tensor: 每个样本的 LLFC 损失，形状为 (batch_size, )。

        功能:
            1. 确认模型数量和批次大小。
            2. 确认权重总和在允许的误差范围内。
            3. 计算加权和输出：将每个模型的输出乘以对应权重并累加。
            4. 应用激活函数（如 GELU）确保输出为非负。
            5. 计算所有模型输出的加权平均值（模拟 MOE 输出）。
            6. 计算加权和输出与加权平均输出之间的平方差的2-范数，得到每个样本的 LLFC 损失。
        """
        num_models = len(layer_outputs)  # 模型数量
        batch_size = layer_outputs[0].size(0)  # 批次大小，假设所有模型的输出批次大小相同
        seq_length, hidden_dim = layer_outputs[0].size(1), layer_outputs[0].size(2)  # 序列长度和隐藏维度

        # 2. 确认权重总和在允许的误差范围内
        total_weight = sum(weights)
        tolerance = 1e-4  # 允许的误差范围
        if not math.isclose(total_weight, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Weights must sum to 1 within a tolerance of {tolerance}. "
                f"Current sum is {total_weight}."
            )

        # 1. 计算加权和输出
        # 形状: (batch_size, seq_length, hidden_dim)
        weighted_sum = torch.zeros_like(layer_outputs[0])  # 初始化加权和张量
        for output, weight in zip(layer_outputs, weights):
            weighted_sum += output * weight  # 加权累加每个模型的输出

        # 3. 应用激活函数，确保输出为非负
        activated_weighted_sum = self.act(weighted_sum)  # 应用激活函数到加权和

        # 4. 计算所有模型输出的加权平均值（模拟 MOE 输出）
        # 为避免内存消耗，逐步累加激活后的输出
        activated_sum = torch.zeros_like(layer_outputs[0])  # 初始化累加张量
        for output, weight in zip(layer_outputs, weights):
            activated_output = self.act(output)  # 应用激活函数到每个模型的输出
            activated_sum += activated_output * weight  # 加权累加

        # 计算加权平均值（已经确认总权重为1）
        average_activated_output = activated_sum  # 形状: (batch_size, seq_length, hidden_dim)

        # 5. 计算加权和输出与加权平均输出之间的差异
        difference = average_activated_output - activated_weighted_sum  # 形状: (batch_size, seq_length, hidden_dim)

        # 6. 计算每个样本的 LLFC 损失
        # 计算每个样本的2-范数
        # torch.norm 支持指定维度和范数类型
        # 这里我们计算每个样本在 (seq_length, hidden_dim) 维度上的2-范数
        sample_losses = torch.norm(difference.view(batch_size, -1), p=2, dim=1)  # 形状: (batch_size, )

        # 可选的详细损失计算信息打印（已注释）
        # print("\nLLFC Loss Calculation Details:")
        # print(f"Original outputs shapes: {[o.shape for o in layer_outputs]}")
        # print(f"Weighted sum shape: {weighted_sum.shape}")
        # print(f"Activated weighted sum shape: {activated_weighted_sum.shape}")
        # print(f"Average activated output shape: {average_activated_output.shape}")
        # print(f"Difference shape: {difference.shape}")
        # print(f"Sample losses shape: {sample_losses.shape}")

        return sample_losses  # 返回包含每个样本 LLFC 损失的张量，形状为 (batch_size, )
    def apply_activation(self, x: torch.Tensor, activation_type: str, params: Dict) -> torch.Tensor:
        """应用激活函数

        Args:
            x (torch.Tensor): 输入张量
            activation_type (str): 激活函数类型，例如 'relu' 或 'gelu'
            params (Dict): 激活函数的参数

        Returns:
            torch.Tensor: 应用激活函数后的张量
        """
        if activation_type == 'relu':
            return torch.nn.functional.relu(x, inplace=params.get('inplace', False))
        elif activation_type == 'gelu':
            return torch.nn.functional.gelu(x,approximate = "tanh")
        return x


    @torch.no_grad()
    def compute_llfc_loss_by_row(
        self,
        sample_input,
        models,  #第一个为pretrain basemodel，后面的是sourcemodel
        weights: List[float],
        layer_name: str
    ):
        """
        计算多个模型在指定层的理论 LLFC 损失，按每个隐藏维度（行）计算一致性损失，并考虑激活函数的影响。
        差异定义为 sigma(avg(Wx)) - avg(sigma(Wx))，并对每个隐藏维度的差异进行平方以确保其非负性。

        Args:
            layer_outputs (List[torch.Tensor]):
                每个模型在目标层的输出张量列表。每个元素的形状为 (batch_size, patch_size, hidden_dim)。
            weights (List[float]):
                每个模型在加权和中的权重列表。长度应与 layer_outputs 相同，且权重总和接近 1。
            layer_name (str):
                要分析的层的名称，用于获取激活函数信息。

        Returns:
            Tensor[float]:
                每个隐藏维度的 LLFC 损失列表，长度为 hidden_dim。每个元素表示该维度上模型输出的一致性损失。
        """
        # 检查权重和输出列表长度是否匹配
        base_model = models[0]  # 选择第一个模型作为基准模型
#这边最好写一个 get_layer_input_and_stop
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)  # 获取目标层的输入张量

            # 获取其他模型在目标层的输出
        layer_outputs = []
        for model in models[1:]:
            output = self.feed_layer(model, layer_name, layer_input)  # 将基准模型的层输入传递给目标层，获取输出
            layer_outputs.append(output)  # 将输出添加到列表中
        if len(weights) != len(layer_outputs):
            raise ValueError("The number of weights must match the number of layer outputs.")

        num_models = len(layer_outputs)

        # 获取层的激活函数信息
       
        activation_type = 'gelu'
        activation_params = {'approximate': 'none'}

        # 如果没有激活函数，直接返回1的列表
        if activation_type == 'none':
            hidden_dim = layer_outputs[0].shape[-1]
            return [1.0] * hidden_dim

        # 将所有模型的输出堆叠成一个张量 (num_models, batch_size, patch_size, hidden_dim)
        stacked_outputs = torch.stack(layer_outputs, dim=0)  # Shape: (num_models, batch_size, patch_size, hidden_dim)
        num_models, batch_size, patch_size, hidden_dim = stacked_outputs.shape

        # 展平为 (num_models, batch_size * patch_size, hidden_dim) 以便批量处理
        flattened_outputs = stacked_outputs.view(num_models, -1, hidden_dim)  # Shape: (num_models, batch_size * patch_size, hidden_dim)

        # 转换权重为张量并调整维度以便广播 (num_models, 1, 1)
        weights_tensor = torch.tensor(weights, device=flattened_outputs.device).view(num_models, 1, 1)  # Shape: (num_models, 1, 1)

        # 计算加权平均 Wx: sum(weight_i * Wx_i)
        weighted_avg_Wx = (flattened_outputs * weights_tensor).sum(dim=0)  # Shape: (batch_size * patch_size, hidden_dim)

        # 应用激活函数到加权平均后的 Wx: sigma(avg(Wx))
        a = self.apply_activation(weighted_avg_Wx, activation_type, activation_params)  # Shape: (batch_size * patch_size, hidden_dim)

        # 应用激活函数到每个模型的 Wx: sigma(Wx_i)
        activated_Wx = self.apply_activation(flattened_outputs, activation_type, activation_params)  # Shape: (num_models, batch_size * patch_size, hidden_dim)

        # 计算加权平均后的 sigma(Wx_i): avg(sigma(Wx))
        weighted_avg_activated_Wx = (activated_Wx * weights_tensor).sum(dim=0)  # Shape: (batch_size * patch_size, hidden_dim)

        # 计算差异: sigma(avg(Wx)) - avg(sigma(Wx))
        diff = a - weighted_avg_activated_Wx  # Shape: (batch_size * patch_size, hidden_dim)

        # 根据激活函数类型调整差异（例如，对于 ReLU，仅在激活输出大于零时考虑差异）
        if activation_type == 'relu':
            # 对于 ReLU，仅考虑输出值为正的区域的差异
            mask = (a > 0) | (weighted_avg_activated_Wx > 0)  # Shape: (batch_size * patch_size, hidden_dim)
            diff = diff * mask  # 仅保留正值区域的差异

        # 计算平方差
        squared_diff = diff ** 2  # Shape: (batch_size * patch_size, hidden_dim)

        # 在批次维度上求平均，得到每个隐藏维度的损失
        dim_loss = squared_diff.mean(dim=0)  # Shape: (hidden_dim,)

        # 转换为列表并返回
        return dim_loss
    
    @torch.no_grad()
    def compute_norm_llfc_loss(self, layer_outputs: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        """
        计算多个模型在指定层的规范化 LLFC 损失，衡量它们在该层输出的一致性，并使损失对层输出的缩放不敏感。
        
        Args:
            layer_outputs (List[torch.Tensor]): 每个模型在目标层的输出张量列表。
                每个元素的形状为 (batch_size, seq_length, hidden_dim)。
            weights (List[float]): 每个模型在加权和中的权重列表。
                长度应与 layer_outputs 相同，且权重总和接近 1。
        
        Returns:
            torch.Tensor: 每个样本的规范化 LLFC 损失，形状为 (batch_size, )。
        
        功能:
            1. 确认模型数量和批次大小。
            2. 确认权重总和在允许的误差范围内。
            3. 计算加权和输出：将每个模型的输出乘以对应权重并累加。
            4. 应用激活函数（如 GELU）确保输出为非负。
            5. 计算所有模型输出的加权平均值（模拟 MOE 输出）。
            6. 计算加权和输出与加权平均输出之间的差异的2-范数，并进行归一化，得到每个样本的规范化 LLFC 损失。
        """
        num_models = len(layer_outputs)  # 模型数量
        batch_size = layer_outputs[0].size(0)  # 批次大小，假设所有模型的输出批次大小相同
        seq_length, hidden_dim = layer_outputs[0].size(1), layer_outputs[0].size(2)  # 序列长度和隐藏维度

        # 2. 确认权重总和在允许的误差范围内
        total_weight = sum(weights)
        tolerance = 1e-4  # 允许的误差范围
        if not math.isclose(total_weight, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Weights must sum to 1 within a tolerance of {tolerance}. "
                f"Current sum is {total_weight}."
            )

        # 1. 计算加权和输出
        # 形状: (batch_size, seq_length, hidden_dim)
        weighted_sum = torch.zeros_like(layer_outputs[0])  # 初始化加权和张量
        for output, weight in zip(layer_outputs, weights):
            weighted_sum += output * weight  # 加权累加每个模型的输出

        # 3. 应用激活函数，确保输出为非负
        activated_weighted_sum = self.act(weighted_sum)  # 应用激活函数到加权和

        # 4. 计算所有模型输出的加权平均值（模拟 MOE 输出）
        # 为避免内存消耗，逐步累加激活后的输出
        activated_sum = torch.zeros_like(layer_outputs[0])  # 初始化累加张量
        for output, weight in zip(layer_outputs, weights):
            activated_output = self.act(output)  # 应用激活函数到每个模型的输出
            activated_sum += activated_output * weight  # 加权累加

        # 计算加权平均值（已经确认总权重为1）
        average_activated_output = activated_sum  # 形状: (batch_size, seq_length, hidden_dim)

        # 5. 计算加权和输出与加权平均输出之间的差异
        difference = average_activated_output - activated_weighted_sum  # 形状: (batch_size, seq_length, hidden_dim)

        # 计算 norm_a (2-范数) 对于 activated_weighted_sum
        norm_a = torch.norm(activated_weighted_sum.view(batch_size, -1), p=2, dim=1, keepdim=True)  # 形状: (batch_size, 1)

        # 防止除以零
        epsilon = 1e-8
        norm_a = norm_a + epsilon

        # 6. 规范化 difference
        normalized_difference = difference / norm_a.unsqueeze(-1)  # 形状: (batch_size, seq_length, hidden_dim)

        # 计算每个样本的 LLFC 损失（2-范数）
        # 计算每个样本的2-范数
        sample_losses = torch.norm(normalized_difference.view(batch_size, -1), p=2, dim=1)  # 形状: (batch_size, )

        return sample_losses  # 返回包含每个样本规范化 LLFC 损失的张量，形状为 (batch_size, )



    @torch.no_grad()
    def compute_llfc_loss_without_merge(self, layer_outputs: List[torch.Tensor], weights: List[float],base_model_output) -> torch.Tensor:
        #base_model: the merged model
        """
        计算多个模型在指定层的 LLFC 损失，衡量它们在该层输出的一致性。

        Args:
            layer_outputs (List[torch.Tensor]): 每个模型在目标层的输出张量列表。
                每个元素的形状为 (batch_size, seq_length, hidden_dim)。
            weights (List[float]): 每个模型在加权和中的权重列表。
                长度应与 layer_outputs 相同，且权重总和接近 1。

        Returns:
            torch.Tensor: 每个样本的 LLFC 损失，形状为 (batch_size, )。

        功能:
            1. 确认模型数量和批次大小。
            2. 确认权重总和在允许的误差范围内。
            3. 计算加权和输出：将每个模型的输出乘以对应权重并累加。
            4. 应用激活函数（如 GELU）确保输出为非负。
            5. 计算所有模型输出的加权平均值（模拟 MOE 输出）。
            6. 计算加权和输出与加权平均输出之间的平方差的2-范数，得到每个样本的 LLFC 损失。
        """
        num_models = len(layer_outputs)  # 模型数量
        batch_size = layer_outputs[0].size(0)  # 批次大小，假设所有模型的输出批次大小相同
        seq_length, hidden_dim = layer_outputs[0].size(1), layer_outputs[0].size(2)  # 序列长度和隐藏维度

        # 2. 确认权重总和在允许的误差范围内
        total_weight = sum(weights)
        tolerance = 1e-4  # 允许的误差范围
        if not math.isclose(total_weight, 1.0, abs_tol=tolerance):
            raise ValueError(
                f"Weights must sum to 1 within a tolerance of {tolerance}. "
                f"Current sum is {total_weight}."
            )

        # 1. 计算加权和输出
        # 形状: (batch_size, seq_length, hidden_dim)
        weighted_sum = base_model_output

        # 3. 应用激活函数，确保输出为非负
        activated_weighted_sum = self.act(weighted_sum)  # 应用激活函数到加权和

        # 4. 计算所有模型输出的加权平均值（模拟 MOE 输出）
        # 为避免内存消耗，逐步累加激活后的输出
        activated_sum = torch.zeros_like(layer_outputs[0])  # 初始化累加张量
        for output, weight in zip(layer_outputs, weights):
            activated_output = self.act(output)  # 应用激活函数到每个模型的输出
            activated_sum += activated_output * weight  # 加权累加

        # 计算加权平均值（已经确认总权重为1）
        average_activated_output = activated_sum  # 形状: (batch_size, seq_length, hidden_dim)

        # 5. 计算加权和输出与加权平均输出之间的差异
        difference = average_activated_output - activated_weighted_sum  # 形状: (batch_size, seq_length, hidden_dim)

        # 6. 计算每个样本的 LLFC 损失
        # 计算每个样本的2-范数
        # torch.norm 支持指定维度和范数类型
        # 这里我们计算每个样本在 (seq_length, hidden_dim) 维度上的2-范数
        sample_losses = torch.norm(difference.view(batch_size, -1), p=2, dim=1)  # 形状: (batch_size, )

        # 可选的详细损失计算信息打印（已注释）
        # print("\nLLFC Loss Calculation Details:")
        # print(f"Original outputs shapes: {[o.shape for o in layer_outputs]}")
        # print(f"Weighted sum shape: {weighted_sum.shape}")
        # print(f"Activated weighted sum shape: {activated_weighted_sum.shape}")
        # print(f"Average activated output shape: {average_activated_output.shape}")
        # print(f"Difference shape: {difference.shape}")
        # print(f"Sample losses shape: {sample_losses.shape}")

        return sample_losses  # 返回包含每个样本 LLFC 损失的张量，形状为 (batch_size, )





    #在前几层都merge完了之后计算layer loss


#最好加上and stop以减少inference cost
    def analyze_layer(
            self,
            layer_name: str,
            models: List[nn.Module],
            weights: List[float],
            sample_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        分析指定层的 LLFC 损失（新版方法）。

        Args:
            layer_name (str): 要分析的层名称。
            models (List[nn.Module]): 模型列表，第一个为基准模型,用于get layer input
            weights (List[float]): 每个模型在损失计算中的权重列表。长度应与 models 相同。
            sample_input (torch.Tensor): 输入张量，形状为 (batch_size, ...)，与模型输入匹配。

        Returns:
            torch.Tensor: 包含所有样本 LLFC 损失的张量，形状为 (batch_size, )。

        功能:
            1. 获取基准模型（第一个模型）在目标层的输入张量。
            2. 将基准模型的层输入传递给其他模型的相同层，获取这些模型在该层的输出。
            3. 计算这些模型输出的 LLFC 损失，基于指定的权重。
            4. 返回所有样本的 LLFC 损失。
        """
        all_losses = []  # 初始化存储所有损失的列表

        try:
            # 获取基准模型（第一个模型）在目标层的输入
            base_model = models[0]  # 选择第一个模型作为基准模型
#这边最好写一个 get_layer_input_and_stop
            layer_input = self.get_layer_input(base_model, layer_name, sample_input)  # 获取目标层的输入张量

            # 获取其他模型在目标层的输出
            layer_outputs = []
            for model in models[1:]:
                output = self.feed_layer(model, layer_name, layer_input)  # 将基准模型的层输入传递给目标层，获取输出
                layer_outputs.append(output)  # 将输出添加到列表中

            # 计算 LLFC 损失
            batch_losses = self.compute_llfc_loss(layer_outputs, weights[1:])  # 传入除基准模型外的权重
            all_losses.append(batch_losses)  # 将损失添加到列表中

            # 返回所有损失的拼接结果

#这个拼接没有意义？all losses只有一个元素
            return torch.cat(all_losses)  # 返回形状为 (batch_size, ) 的张量

        except Exception as e:
            print(f"Error in analyze_layer: {e}")  # 打印错误信息
            raise  # 重新抛出异常以便调用者处理

    def analyze_layer_without_merge(
            self,
            layer_name: str,
            models: List[nn.Module],
            weights: List[float],
            sample_input: torch.Tensor,
            base_model: nn.Module # the merged model
    ) -> torch.Tensor:
        """
        分析指定层的 LLFC 损失，不使用基准模型的层输入进行比较。

        Args:
            layer_name (str): 要分析的层名称。
            models (List[nn.Module]): 模型列表，第一个为基准模型，后续模型与基准模型进行比较。
            weights (List[float]): 每个模型在损失计算中的权重列表。长度应与 models 相同。
            sample_input (torch.Tensor): 输入张量，形状为 (batch_size, ...)，与模型输入匹配。

        Returns:
            torch.Tensor: 包含所有样本 LLFC 损失的张量，形状为 (batch_size, )。

        功能:
            1. 遍历所有模型，直接获取它们在目标层的输出。
            2. 计算这些模型输出的 LLFC 损失，基于指定的权重。
            3. 返回所有样本的 LLFC 损失。
        """
        all_losses = []  # 初始化存储所有损失的列表

        try:
            layer_outputs = []
            for model in models:
                # 获取每个模型在目标层的输出
                output = self.get_intermediate_output(model, layer_name, sample_input)
                layer_outputs.append(output)  # 将输出添加到列表中
            base_output = self.get_intermediate_output(base_model, layer_name, sample_input)
            # 计算 LLFC 损失
            batch_losses = self.compute_llfc_loss_without_merge(layer_outputs, weights,base_output)  # 传入所有模型的权重
            all_losses.append(batch_losses)  # 将损失添加到列表中

            # 返回所有损失的拼接结果
            return torch.cat(all_losses)  # 返回形状为 (batch_size, ) 的张量

        except Exception as e:
            print(f"Error in analyze_layer: {e}")  # 打印错误信息
            raise  # 重新抛出异常以便调用者处理

    def analyze_layer_output(
            self,
            layer_name: str,
            models: List[nn.Module],
            sample_input: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        返回基准模型以外的模型在指定层的输出列表。

        Args:
            layer_name (str): 层的名称。
            models (List[nn.Module]): 模型列表，第一个模型为基准模型，其余为微调模型。
            sample_input (torch.Tensor): 输入样本张量，形状为 (batch_size, ...)，与模型输入匹配。

        Returns:
            List[torch.Tensor]: 微调模型在指定层的输出列表，每个元素形状为 (batch_size, ...)

        功能:
            1. 获取基准模型（第一个模型）在目标层的输入张量。
            2. 将基准模型的层输入传递给其他模型的相同层，获取这些模型在该层的输出。
            3. 返回所有微调模型在目标层的输出列表。
        """
        # 获取基准模型的层输入
        base_model = models[0]  # 选择第一个模型作为基准模型
        layer_input = self.get_layer_input(base_model, layer_name, sample_input)  # 获取目标层的输入张量

        # 计算微调模型在该层的输出
        layer_outputs = []
        for model in models[1:]:
            output = self.feed_layer(model, layer_name, layer_input)  # 将层输入传递给目标层，获取输出
            layer_outputs.append(output)  # 将输出添加到列表中

        return layer_outputs  # 返回所有微调模型在目标层的输出列表

    def get_intermediate_output(
            self,
            model: nn.Module,
            target_layer: str,
            input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        获取模型在指定层的输出，并在该层后停止前向传播。

        Args:
            model (nn.Module): 要分析的模型。
            target_layer (str): 目标层的名称。
            input_tensor (torch.Tensor): 输入张量，形状为 (batch_size, ...)，与模型输入匹配。

        Returns:
            torch.Tensor: 目标层的输出张量，形状与目标层输出匹配。

        功能:
            1. 注册单层钩子，并在目标层后停止前向传播。
            2. 将输入张量传递给模型，触发钩子，捕获目标层的输出并停止传播。
            3. 清除所有钩子，确保不会影响后续操作。
            4. 返回目标层的输出张量。
            5. 如果未能成功获取输出，抛出异常。
        """
        self.register_single_hook_and_stop(model, target_layer)  # 注册钩子，并在目标层后停止前向传播

        try:
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 禁用梯度计算
                _ = model(input_tensor)  # 执行前向传播，触发钩子和异常
        except StopForwardException as e:
            return self.activation_outputs[target_layer]  # 捕获异常后，返回目标层的输出
        finally:
            self.clear_hooks()  # 移除所有钩子，确保不会影响后续操作

        # 如果未能成功获取输出，抛出异常
        raise ValueError(f"Failed to get output for layer {target_layer}")
