import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from typing import Tuple, Dict, List
import logging
from logging import Logger

# Set up logging
logger: Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VisionTransformerBlock(nn.Module):
    """
    A single block of the Vision Transformer model.

    Args:
    - num_heads (int): The number of attention heads.
    - hidden_size (int): The size of the hidden state.
    - dropout (float): The dropout probability.
    """

    def __init__(self, num_heads: int, hidden_size: int, dropout: float):
        super(VisionTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor.
        """
        x = x + self.attn(self.norm1(x), self.norm1(x), value=self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    """
    The Vision Transformer encoder model.

    Args:
    - num_layers (int): The number of layers.
    - num_heads (int): The number of attention heads.
    - hidden_size (int): The size of the hidden state.
    - dropout (float): The dropout probability.
    """

    def __init__(self, num_layers: int, num_heads: int, hidden_size: int, dropout: float):
        super(VisionTransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([VisionTransformerBlock(num_heads, hidden_size, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor.
        """
        for block in self.blocks:
            x = block(x)
        return x


class VisionTransformerDecoder(nn.Module):
    """
    The Vision Transformer decoder model.

    Args:
    - num_layers (int): The number of layers.
    - num_heads (int): The number of attention heads.
    - hidden_size (int): The size of the hidden state.
    - dropout (float): The dropout probability.
    """

    def __init__(self, num_layers: int, num_heads: int, hidden_size: int, dropout: float):
        super(VisionTransformerDecoder, self).__init__()
        self.blocks = nn.ModuleList([VisionTransformerBlock(num_heads, hidden_size, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor.
        """
        for block in self.blocks:
            x = block(x)
        return x


class VisionTransformerModel(nn.Module):
    """
    The Vision Transformer model.

    Args:
    - num_layers (int): The number of layers.
    - num_heads (int): The number of attention heads.
    - hidden_size (int): The size of the hidden state.
    - dropout (float): The dropout probability.
    """

    def __init__(self, num_layers: int, num_heads: int, hidden_size: int, dropout: float):
        super(VisionTransformerModel, self).__init__()
        self.encoder = VisionTransformerEncoder(num_layers, num_heads, hidden_size, dropout)
        self.decoder = VisionTransformerDecoder(num_layers, num_heads, hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ModelArchitecture(nn.Module):
    """
    The model architecture.

    Args:
    - num_layers (int): The number of layers.
    - num_heads (int): The number of attention heads.
    - hidden_size (int): The size of the hidden state.
    - dropout (float): The dropout probability.
    """

    def __init__(self, num_layers: int, num_heads: int, hidden_size: int, dropout: float):
        super(ModelArchitecture, self).__init__()
        self.model = VisionTransformerModel(num_layers, num_heads, hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor.
        """
        return self.model(x)

    def build_transformer_block(self, num_heads: int, hidden_size: int, dropout: float) -> VisionTransformerBlock:
        """
        Builds a Vision Transformer block.

        Args:
        - num_heads (int): The number of attention heads.
        - hidden_size (int): The size of the hidden state.
        - dropout (float): The dropout probability.

        Returns:
        - VisionTransformerBlock: The Vision Transformer block.
        """
        return VisionTransformerBlock(num_heads, hidden_size, dropout)


def main():
    # Example usage
    num_layers = 6
    num_heads = 8
    hidden_size = 512
    dropout = 0.1
    model = ModelArchitecture(num_layers, num_heads, hidden_size, dropout)
    input_tensor = torch.randn(1, hidden_size)
    output = model(input_tensor)
    print(output.shape)


if __name__ == "__main__":
    main()