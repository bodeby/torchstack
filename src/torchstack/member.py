from transformers import AutoModelForCausalLM
import torch
import math


class AutoModelMember(AutoModelForCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.weight = -math.inf
        self.current_device = "cpu"  # Default to CPU


    # verify is member weight is valid
    def _is_valid_weight(self, weight: float):
        if not (0.0 <= weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0")
        return weight

    # Move model to a specified device
    def _offload_to_gpu(self, device: str) -> None:
        """
        Move the model to the specified device.
        Args:
            device (str): Target device ('cpu', 'cuda', 'cuda:<index>', or 'mps').
        """
        self.to(device)
        self.current_device = device

    # Efficiently offload the model to CPU
    def _offload_to_cpu(self) -> None:
        """
        Offloads the model to CPU to free GPU memory.
        """
        self.move_to("cpu")

    # Prepare model for inference on optimal device
    def _prepare_for_inference(self) -> None:
        """
        Prepares the model on the appropriate hardware for inference.
        Chooses CUDA if available, otherwise falls back to MPS or CPU.
        """
        if torch.cuda.is_available():
            self._offload_to_gpu()
        elif torch.backends.mps.is_available():
            self._offload_to_gpu()
        else:
            self._offload_to_cpu()

    # add model weights

    # add weights
    def modify_weight(self, weight: float):
        if self._is_valid_weight(weight):
            self.weight = weight

    # Generate tokens while ensuring device readiness
    def generate_token(self, inputs, **generate_kwargs):
        """
        Generate tokens using the model, ensuring it is moved to the active device.
        Args:
            inputs: Input tensor.
            **generate_kwargs: Additional arguments for the generate method.
        """
        if self.current_device != "cuda:0":
            self._prepare_for_inference()

        outputs = self.generate(inputs, **generate_kwargs)
        return outputs
