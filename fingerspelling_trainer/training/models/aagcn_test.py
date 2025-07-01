import torch
from torch import nn
from typing import Dict
from omegaconf import DictConfig

from fingerspelling_trainer.training.utils.graph_utils import (
    EdgeIndexBuilder,
    LandmarkType,
)
from torch_geometric_temporal.nn.attention.tsagcn import (
    AAGCN as _AAGCN,
    UnitGCN as _UnitGCN,
)


class _SafeUnitGCN(_UnitGCN):
    def _adaptive_forward(self, x, y):
        N, C, T, V = x.size()
        A = self.PA
        for i in range(self.num_subset):
            A1 = (
                self.conv_a[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.conv_b[i](x).reshape(N, self.inter_c * T, V)
            A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))
            A1 = A[i] + A1 * self.alpha
            A2 = x.contiguous().view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y

    def _non_adaptive_forward(self, x, y):
        N, C, T, V = x.size()
        for i in range(self.num_subset):
            A1 = self.A[i]
            A2 = x.contiguous().view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y


class _SafeAAGCN(_AAGCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn1 = _SafeUnitGCN(
            self.gcn1.in_c,
            self.gcn1.out_c,
            self.gcn1.A,  # type: ignore
            adaptive=self.gcn1.adaptive,
            attention=self.gcn1.attention,
        )


class FingerPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1, bias=False)
        M = torch.zeros(21, 5)
        for k, idx in enumerate(
            [range(1, 5), range(5, 9), range(9, 13), range(13, 17), range(17, 21)]
        ):
            M[idx, k] = 1.0
        self.register_buffer("M", M)

    def forward(self, h):
        vel, core = h[:, 21:22], h[:, :21]
        alpha = torch.softmax(self.score(core), dim=1)
        dedos = torch.einsum("bnh,nk->bkh", alpha * core, self.M)
        out = torch.cat([dedos, vel], dim=1)
        return out.reshape(h.size(0), -1)


class LiteTCN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 96, depth: int = 4, k: int = 5):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv1d(
                    in_dim if i == 0 else hid,
                    hid,
                    k,
                    padding=((k - 1) // 2) * (2**i),
                    dilation=2**i,
                ),
                nn.ReLU(),
            ]
        self.net = nn.Sequential(*layers)
        self.out_dim = hid

    def forward(self, x):
        x = self.net(x.transpose(1, 2))
        return x.transpose(1, 2)


class StackedAAGCN(nn.Module):
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, cfg: DictConfig):
        super().__init__()
        dims = cfg.hidden_dims
        in_c = 3
        layers = []
        for h in dims:
            layers.append(
                _SafeAAGCN(
                    in_channels=in_c,
                    out_channels=h,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    stride=1,
                    residual=True,
                    adaptive=True,
                    attention=True,
                )
            )
            in_c = h
        self.layers = nn.ModuleList(layers)
        self.out_dim = in_c

    def forward(self, x):
        for g in self.layers:
            x = g(x)
        return x


class AAGCNTranslator(nn.Module):
    """
    Node 22 = wrist velocity.
    AAGCN ==> (finger-pool OR flatten) ==> LiteTCN ==> MLP ==> logits for CTC
    """

    def __init__(self, model_cfg: DictConfig, vocab_size: int):
        super().__init__()
        builder: EdgeIndexBuilder = model_cfg.gnn.edge_index_builder

        ei_base = builder.build()[LandmarkType.LEFT_HAND]
        ei = builder.add_velocity_node(ei_base, strategy=model_cfg.gnn.vel_conn)
        num_nodes = 22

        self.gnn = StackedAAGCN(ei, num_nodes=num_nodes, cfg=model_cfg.gnn)
        gnn_H = self.gnn.out_dim

        if model_cfg.spatial_pool == "finger":
            self.pool = FingerPool(gnn_H)
            feat_dim = gnn_H * 6
        elif model_cfg.spatial_pool == "none":
            self.pool = None
            feat_dim = gnn_H * num_nodes
        else:
            raise ValueError("spatial_pool ∈ {'none','finger'}")

        self.tcn = LiteTCN(
            feat_dim,
            hid=model_cfg.tcn.hidden_dim,
            depth=model_cfg.tcn.depth,
            k=model_cfg.tcn.kernel_size,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.tcn.out_dim),
            nn.Linear(self.tcn.out_dim, model_cfg.mlp.hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.mlp.dropout),
            nn.Linear(model_cfg.mlp.hidden_dim, vocab_size),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        rh, lh = batch["right"], batch["left"]
        mask, kp_r, kp_l = batch["mask"], batch["kp_mask_right"], batch["kp_mask_left"]
        rh, lh = rh * kp_r.unsqueeze(-1), lh * kp_l.unsqueeze(-1)
        hand = torch.where(batch["signing_hands"].view(-1, 1, 1, 1) == 1, rh, lh)
        hand = hand * mask.unsqueeze(-1).unsqueeze(-1)  # [B,S,21,3]
        vel_node = batch["wrist_vel"].unsqueeze(2)  # [B,S,1,3]
        hand = torch.cat([hand, vel_node], dim=2)  # [B,S,22,3]
        B, S, _, _ = hand.shape
        x = hand.permute(0, 3, 1, 2)  # [B,3,S,22]
        gnn_out = self.gnn(x).permute(0, 2, 1, 3).contiguous()  # [B,S,H,22]
        if self.pool is None:
            feats = gnn_out.reshape(B, S, -1)
        else:
            feats = self.pool(gnn_out.reshape(B * S, 22, -1)).view(B, S, -1)
        feats = self.tcn(feats)
        logits = self.head(feats)
        lengths = mask.sum(1).cpu()
        return logits, lengths
