from typing import Dict
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from omegaconf import DictConfig

from fingerspelling_trainer.training.utils.graph_utils import (
    EdgeIndexBuilder,
    LandmarkType,
)
from torch_geometric.nn import EdgeConv


class FingerPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1, bias=False)
        M = torch.zeros(21, 5)
        for k, idxs in enumerate(
            [range(1, 5), range(5, 9), range(9, 13), range(13, 17), range(17, 21)]
        ):
            M[idxs, k] = 1.0
        self.register_buffer("M", M)

    def forward(self, h):
        alpha = torch.softmax(self.score(h), dim=1)
        h = alpha * h
        fingers = torch.einsum("bnh,nk->bkh", h, self.M)
        return fingers.reshape(h.size(0), -1)  # [B, 5*H]


class EdgeConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25):
        super().__init__()
        self.edge_conv = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(in_dim * 2, out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            ),
            aggr="mean",
        )
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.LeakyReLU(inplace=True)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        h = self.edge_conv(x, ei)
        h = self.norm(h)
        if self.in_dim == self.out_dim:
            h = h + x
        return self.act(h)


class GNNModule(nn.Module):
    def __init__(self, edge_index: torch.Tensor, cfg: DictConfig):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        dims = [cfg.input_dim] + cfg.hidden_dims
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(
                EdgeConvBlock(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    dropout=cfg.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B*S, N, C]
        bsn, n, _ = x.shape
        x = x.view(-1, x.shape[-1])  # [B*S*N, C]
        batch_ei = self.edge_index.repeat(1, bsn)  # type: ignore
        offset = (
            torch.arange(bsn, device=x.device).repeat_interleave(self.edge_index.size(1)) * n  # type: ignore
        )
        batch_ei = batch_ei + offset
        for blk in self.blocks:
            x = blk(x, batch_ei)
        return x.view(bsn, n, self.out_dim)  # [B*S, N, H]


class EdgeConvTranslator(nn.Module):
    def __init__(self, model_cfg: DictConfig, vocab_size: int):
        super().__init__()
        self.print_name()
        self.model_cfg = model_cfg

        builder_instance = model_cfg.gnn.edge_index_builder
        if not isinstance(builder_instance, EdgeIndexBuilder):
            raise TypeError(
                f"Expected model_cfg.gnn.edge_index_builder to be an instance of EdgeIndexBuilder, "
                f"but got {type(builder_instance)}."
            )
        ei_dict = builder_instance.build()
        ei = ei_dict[LandmarkType.RIGHT_HAND]
        num_nodes = 21

        # GNN
        self.gnn = GNNModule(ei, model_cfg.gnn)
        gnn_H = self.gnn.out_dim

        # Pooling
        self.spatial_pool = model_cfg.spatial_pool
        if self.spatial_pool == "finger":
            self.pool = FingerPool(gnn_H)
            gnn_feat = gnn_H * 5
        elif self.spatial_pool == "none":
            self.pool = None
            gnn_feat = gnn_H * num_nodes
        else:
            raise ValueError("spatial_pool must be 'none' or 'finger'")

        # add velocity as feature?
        self.add_velocity = model_cfg.add_velocity
        lstm_input_dim = gnn_feat + (3 if self.add_velocity else 0)
        self.norm_concat = nn.LayerNorm(lstm_input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=model_cfg.lstm.hidden_dim,
            num_layers=model_cfg.lstm.num_layers,
            batch_first=True,
            dropout=(
                model_cfg.lstm.rnn_dropout if model_cfg.lstm.num_layers > 1 else 0.0
            ),
            bidirectional=model_cfg.lstm.bidirectional,
        )
        lstm_out = model_cfg.lstm.hidden_dim * (
            2 if model_cfg.lstm.bidirectional else 1
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, model_cfg.mlp.hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.mlp.dropout),
            nn.Linear(model_cfg.mlp.hidden_dim, vocab_size),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        rh = batch["right"]
        lh = batch["left"]
        mask = batch["mask"]
        kp_r = batch["kp_mask_right"]
        kp_l = batch["kp_mask_left"]
        signing_hands = batch["signing_hands"]

        rh = rh * kp_r.unsqueeze(-1)
        lh = lh * kp_l.unsqueeze(-1)
        hands = torch.where(signing_hands.view(-1, 1, 1, 1) == 1, rh, lh)
        hands = hands * mask.unsqueeze(-1).unsqueeze(-1)  # [B,S,N,C]

        B, S, N, C = hands.shape
        gnn_in = hands.view(B * S, N, C)
        gnn_out = self.gnn(gnn_in)  # [B*S, N, H]

        # Pooling
        if self.pool is None:
            feats = gnn_out.view(B, S, -1)
        else:
            feats = self.pool(gnn_out)  # [B*S, 5H]
            feats = feats.view(B, S, -1)  # [B, S, 5H]

        # add velocity if configured
        if self.add_velocity:
            feats = torch.cat([feats, batch["wrist_vel"]], dim=-1)
        feats = self.norm_concat(feats)

        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(
            feats, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=S)

        logits = self.mlp(lstm_out)
        return logits, lengths

    def print_name(self):
        print("\n**********************\n")
        print("=== Model : EdgeConv + BiLSTM ===")
        print("\n**********************\n")
