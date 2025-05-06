import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder network for contrastive learning of state embeddings.

    Maps state (and optionally primitive type one-hot) to a latent embedding.
    """
    def __init__(
        self,
        dim_state: int,
        latent_dim: int,
        hidden_layers: list,
        n_primitives: int = 1,
        multi_motion: bool = False,
        activation: nn.Module = nn.GELU
    ):
        super().__init__()
        self.dim_state = dim_state
        self.latent_dim = latent_dim
        self.multi_motion = multi_motion
        self.n_primitives = n_primitives

        # Activation and normalization
        self.activation = activation()
        self.layer_norms = nn.ModuleList()

        # Input dimension: state + primitive one-hot if multi-motion
        input_dim = dim_state + (n_primitives if multi_motion else 0)

        # Build MLP layers
        dims = [input_dim] + hidden_layers + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # Add LayerNorm after every hidden layer
            if i < len(hidden_layers):
                norm = nn.LayerNorm(dims[i+1])
                self.layer_norms.append(norm)
                layers.append(norm)
                layers.append(self.activation)
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, primitive_type: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Tensor of shape (batch_size, dim_state)
            primitive_type: LongTensor of shape (batch_size,) or one-hot Tensor (batch_size, n_primitives)
        Returns:
            embedding: Tensor of shape (batch_size, latent_dim)
        """
        x = state
        if self.multi_motion:
            # Build one-hot encoding if necessary
            if primitive_type.dtype in (torch.int32, torch.int64):
                one_hot = torch.zeros(x.size(0), self.n_primitives, device=x.device)
                one_hot.scatter_(1, primitive_type.unsqueeze(1), 1.0)
            else:
                one_hot = primitive_type
            x = torch.cat([x, one_hot], dim=1)

        # Pass through MLP
        embedding = self.network(x)
        return embedding
