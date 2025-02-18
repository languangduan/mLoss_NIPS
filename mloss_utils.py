import torch
import random
from torch.utils.data import Subset
from src.datasets.registry import get_dataset

#####################################################################
# 1. SUBSET / SAMPLE A DATASET
#####################################################################

def get_subset_of_dataset(
    dataset_name: str,
    preprocess,
    location: str,
    batch_size: int = 32,
    num_workers: int = 4,
    subset_fraction: float = 0.1,
    max_subset_size: int = None,
    seed: int = 0
):
    """
    Loads the dataset using src.datasets.registry.get_dataset, then returns
    a subset of the training data.

    Args:
        dataset_name (str): Name of the dataset, e.g. 'MNIST', 'CIFAR10', etc.
        preprocess: The transformation/preprocessing used by the dataset.
        location (str): Path to where the dataset is stored.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for the DataLoader.
        subset_fraction (float): Fraction of the dataset to keep (e.g., 0.1 for 10%).
        max_subset_size (int, optional): If set, caps the subset size.
        seed (int): Random seed for reproducibility.

    Returns:
        A PyTorch DataLoader for the subset of the dataset.
    """
    dataset = get_dataset(
        dataset_name=dataset_name,
        preprocess=preprocess,
        location=location,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    full_dataset = dataset.train_dataset  # or dataset.test_dataset if you want to subset the test set
    full_size = len(full_dataset)
    subset_size = int(full_size * subset_fraction)

    if max_subset_size is not None:
        subset_size = min(subset_size, max_subset_size)

    indices = list(range(full_size))
    random.seed(seed)
    random.shuffle(indices)
    subset_indices = indices[:subset_size]

    subset_dataset = Subset(full_dataset, subset_indices)

    subset_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    return subset_loader

#####################################################################
# 2 & 3. CAPTURING INPUTS / OUTPUTS FOR A SPECIFIC LAYER
#####################################################################

class LayerIOHook:
    """
    A forward hook to capture the input and output of a specific layer.
    Usage:
        hook = LayerIOHook(model, 'block.ln_2')
        # ... forward pass ...
        outputs = hook.outputs   # list of outputs from each forward pass
        inputs = hook.inputs     # list of inputs from each forward pass
    """
    def __init__(self, model: torch.nn.Module, layer_name: str):
        """
        Args:
            model (torch.nn.Module): The PyTorch model (e.g., ViT).
            layer_name (str): The name of the submodule to hook into.
                              Example for CLIP-ViT: "transformer.resblocks.0.mlp"
        """
        self.model = model
        self.layer_name = layer_name
        self.handle = None
        self.inputs = []
        self.outputs = []

        # Register the forward hook
        self._register_hook()

    def _register_hook(self):
        """
        Finds the submodule by name in the model and registers a forward hook.
        """
        submodule = dict(self.model.named_modules()).get(self.layer_name, None)
        if submodule is None:
            raise ValueError(f"Submodule {self.layer_name} not found in the model.")

        self.handle = submodule.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, module_input, module_output):
        """
        Hook function that saves the input and output.
        """
        # module_input and module_output could be tuples or Tensors
        # Convert them to Tensors if needed
        if isinstance(module_input, tuple):
            self.inputs.append([i.detach().cpu() for i in module_input])
        else:
            self.inputs.append(module_input.detach().cpu())

        if isinstance(module_output, tuple):
            self.outputs.append([o.detach().cpu() for o in module_output])
        else:
            self.outputs.append(module_output.detach().cpu())

    def remove(self):
        """
        Removes the forward hook to free resources.
        """
        if self.handle is not None:
            self.handle.remove()

def get_layer_output(model, layer_name, dataloader, device='cuda'):
    """
    Returns the outputs of a given layer for all batches in a dataloader.
    """
    hook = LayerIOHook(model, layer_name)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            # A typical batch might be either (images, labels) or dict
            # Adapt as needed based on your dataset structure
            if isinstance(batch, dict):
                images = batch['images'].to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0].to(device)
            else:
                raise ValueError("Unknown batch format. Expected dict or (images, labels).")

            _ = model(images)

    outputs = hook.outputs  # list of Tensors from each batch forward pass
    hook.remove()
    return outputs

def get_layer_input(model, layer_name, dataloader, device='cuda'):
    """
    Returns the inputs to a given layer for all batches in a dataloader.
    """
    hook = LayerIOHook(model, layer_name)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch['images'].to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0].to(device)
            else:
                raise ValueError("Unknown batch format. Expected dict or (images, labels).")

            _ = model(images)

    inputs = hook.inputs  # list of Tensors from each batch forward pass
    hook.remove()
    return inputs

#####################################################################
# 4. MODIFY A SPECIFIC LAYER'S PARAMETERS
#####################################################################

def set_layer_parameters(model, layer_name, merged_params_dict):
    """
    Modifies a given layer's parameters with merged_params_dict.
    You can pass in something like:
        merged_params_dict = {
            "weight": <torch.Tensor>,
            "bias": <torch.Tensor>
        }
    The function will replace the specified layer's .weight, .bias, etc.
    with the provided Tensors.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_name (str): Name of the layer (must match named_modules()).
        merged_params_dict (dict): Dictionary of parameter name -> Tensor
    """
    submodule = dict(model.named_modules()).get(layer_name, None)
    if submodule is None:
        raise ValueError(f"Submodule {layer_name} not found in the model.")

    # Example: submodule.weight, submodule.bias
    with torch.no_grad():
        for param_name, new_param_value in merged_params_dict.items():
            if not hasattr(submodule, param_name):
                raise ValueError(f"Parameter '{param_name}' not found in {layer_name}.")
            old_param = getattr(submodule, param_name)
            if old_param.shape != new_param_value.shape:
                raise ValueError(
                    f"Shape mismatch for {layer_name}.{param_name}: "
                    f"old={old_param.shape}, new={new_param_value.shape}"
                )
            # Copy the new parameter
            old_param.data.copy_(new_param_value.data)

#####################################################################
# 5. ROW-BY-ROW OUTPUT OF AN FFN LAYER
#####################################################################

def get_ffn_layer_row_outputs(ffn_layer, input_tensor):
    """
    For a given feed-forward layer (FFN) with weight W and bias b,
    compute the output row-by-row. For instance, if y = W x + b,
    the i-th row of W (plus bias[i]) corresponds to y[i].

    This function can handle batch inputs. For a batch of size B:
        - input_tensor is shape: [B, input_dim]
        - W is shape: [output_dim, input_dim]
        - b is shape: [output_dim]
      The i-th row of the output is shape [B], i.e., one scalar per batch item.

    Args:
        ffn_layer (torch.nn.Linear or similar): The FFN layer to inspect.
        input_tensor (torch.Tensor): The batch input [B, input_dim].

    Returns:
        outputs_per_row (torch.Tensor): Shape [output_dim, B], where
            outputs_per_row[i] is the vector of outputs for row i (over the batch).
    """
    if not hasattr(ffn_layer, 'weight') or not hasattr(ffn_layer, 'bias'):
        raise ValueError("ffn_layer must have 'weight' and 'bias' attributes (e.g., torch.nn.Linear).")

    # y = W x^T + b  (in vector form), or y = X @ W^T + b  in PyTorch
    # Here, we do it row by row for clarity, but we can still do it in a single matrix op:
    #   - full_out = input_tensor @ ffn_layer.weight.T + ffn_layer.bias
    # Then we can rearrange it so that the shape is (output_dim, B).

    weight = ffn_layer.weight  # [output_dim, input_dim]
    bias = ffn_layer.bias      # [output_dim]

    # Full layer output: shape [B, output_dim]
    full_output = input_tensor.mm(weight.t()) + bias  # mm is matrix multiply

    # We want shape [output_dim, B] so each row i is the output for row i across the batch
    outputs_per_row = full_output.transpose(0, 1).contiguous()

    return outputs_per_row
