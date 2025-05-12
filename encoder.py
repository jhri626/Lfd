import torch
from typing import Union


class Encoder(torch.nn.Module):
    """
    Task-space → latent-space encoder compatible with existing DataPreprocessor pipeline.

    * **Input** : state x_t (B, dim_state), primitive_type ids or one-hot (B,) or (B, n_primitives)
    * **Output**: latent y_t (B, latent_space_dim)
    """

    def __init__(
        self,
        dim_state: int,
        n_primitives: int,
        latent_space_dim: int = 64,
        hidden_size: int = 256,
        device: Union[str, torch.device] = "cuda",
        multi_motion: bool =False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dim_state = dim_state
        self.n_primitives = n_primitives
        self.latent_space_dim = latent_space_dim
        self.multi_motion = multi_motion

        # One-hot primitive encodings for multi-motion
        self.register_buffer(
            "primitives_encodings", torch.eye(n_primitives, device=self.device)
        )
        self.register_buffer("goals_latent_space", torch.zeros(n_primitives, latent_space_dim, device=device))

        # MLP: [dim_state + n_primitives] → hidden → hidden → latent
        if self.multi_motion == True:
            input_dim = dim_state + n_primitives
        else:
            input_dim = dim_state
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.GELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.GELU(),
        )
        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_size, hidden_size),
        #     torch.nn.LayerNorm(hidden_size),
        #     torch.nn.GELU(),
        # )
        # self.layer4 = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_size, hidden_size),
        #     torch.nn.LayerNorm(hidden_size),
        #     torch.nn.GELU(),
        # )
        
        self.layer5 = torch.nn.Linear(hidden_size, latent_space_dim)

    def encode_primitives(self, primitive_type: torch.Tensor) -> torch.Tensor:
        """
        Convert primitive_type ids or one-hot to one-hot tensor.
        """
        if primitive_type.ndim == 1:
            return self.primitives_encodings[primitive_type.long()]
        elif primitive_type.ndim == 2:
            return primitive_type.to(self.device)
        else:
            raise ValueError("primitive_type must be (B,) ids or (B, n_primitives) one-hot")

    def forward(
        self,
        x_t: torch.Tensor,
        primitive_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: concatenate state + primitive one-hot, apply MLP.
        """
        x = x_t.to(self.device)
        if self.multi_motion == True:
            enc = self.encode_primitives(primitive_type)
            inp = torch.cat([x, enc], dim=1)
        else:
            inp = x
        h = self.layer1(inp)
        h = self.layer2(h)
        # h = self.layer3(h)
        # h = self.layer4(h)
        y_t = self.layer5(h)
        return y_t

    def update_goals_latent_space(self, goals: torch.Tensor):
        """
        Maps task-space goals to latent-space, stores in self.goals_latent_space[i].
        goals: (n_primitives, dim_workspace), normalized.
        """
        for i in range(self.n_primitives):
            prim = torch.tensor([i], device=self.goals_latent_space.device)
            inp = torch.zeros([1, self.dim_state], device=self.goals_latent_space.device)
            inp[:, :goals.shape[1]] = goals[i].to(inp.device)
            self.goals_latent_space[i] = self.forward(inp, prim).squeeze(0).detach()

    def get_goals_latent_space_batch(self, primitive_type: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, latent_space_dim) where each row is goals_latent_space[primitive_type[b]].
        primitive_type: (B,) of integer IDs.
        """
        if primitive_type.ndim == 2:
            primitive_type = primitive_type.argmax(dim=1)
        return self.goals_latent_space[primitive_type.long()]
