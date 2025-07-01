from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMTranslator(nn.Module):
    def __init__(self, model_cfg: DictConfig, vocab_size: int):
        super().__init__()
        self.print_name()
        self.model_cfg = model_cfg
        N = 21
        in_feats = N * 3
        if model_cfg.add_velocity:
            in_feats += 3  # +3 velocity

        self.lstm = nn.LSTM(
            input_size=in_feats,
            hidden_size=model_cfg.lstm.hidden_dim,
            num_layers=model_cfg.lstm.num_layers,
            batch_first=True,
            dropout=(
                model_cfg.lstm.rnn_dropout if model_cfg.lstm.num_layers > 1 else 0.0
            ),
            bidirectional=model_cfg.lstm.bidirectional,
        )

        lstm_output_feature_dim = model_cfg.lstm.hidden_dim * (
            2 if model_cfg.lstm.bidirectional else 1
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                lstm_output_feature_dim,
                model_cfg.mlp.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(model_cfg.mlp.dropout),
            nn.Linear(model_cfg.mlp.hidden_dim, vocab_size),
        )

    def forward(self, batch):
        rh_kps, lh_kps = batch["right"], batch["left"]
        frame_mask, kp_mask_l, kp_mask_r = (
            batch["mask"],
            batch["kp_mask_left"],
            batch["kp_mask_right"],
        )
        signing_hands, lengths = batch["signing_hands"], batch["lengths"]

        kp_mask_r = kp_mask_r.unsqueeze(-1)  # [B,T,21,1]
        kp_mask_l = kp_mask_l.unsqueeze(-1)
        rh_kps *= kp_mask_r
        lh_kps *= kp_mask_l

        mask_expanded = frame_mask.unsqueeze(-1).unsqueeze(-1).float()  # [B,T,1,1]
        selected = (
            torch.where(signing_hands.view(-1, 1, 1, 1).float() == 1, rh_kps, lh_kps)
            * mask_expanded
        )  # [B,T,21,3]

        B, T, N, C = selected.shape
        feats = selected.view(B, T, N * C)

        # Velocity
        if self.model_cfg.add_velocity:
            vel = batch["wrist_vel"]  # [B,T,3]
            feats = torch.cat([feats, vel], dim=-1)

        lengths = lengths.cpu().long()
        if not (lengths > 0).all():
            raise ValueError(
                "Detected at least one sequence with zero length in the batch."
            )

        packed_input = pack_padded_sequence(
            feats, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_output_padded_original_order, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=T
        )

        frame_logits = self.mlp(lstm_output_padded_original_order)
        return frame_logits, lengths

    def print_name(self):
        print("\n**********************\n")
        print("=== Model : LSTM ===")
        print("\n**********************\n")
