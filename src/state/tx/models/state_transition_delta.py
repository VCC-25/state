from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_transition import StateTransitionPerturbationModel
from .decoders_nb import NBDecoder, nb_nll

class DeltaStateTransitionPerturbationModel(StateTransitionPerturbationModel):
"""
Delta-Residual Variante:
final_pred = control_expr + delta_hat
Wobei control_expr = batch["ctrl_cell_emb"] (reshaped) und delta_hat vom Hidden-Feature (res_pred) abgeleitet wird.
Haupt-Loss:   L_main  = loss_fn(pred, target)
Delta-Loss:   L_delta = MSE(delta_hat, target - control_expr)
Gesamt:       L_total = L_main + beta_delta * L_delta + gamma_delta_sparsity * L1(delta_hat) + (optional Decoder + Confidence)

Hinweise:
- predict_residual der Basisklasse wird ignoriert (Delta-Modus setzt immer pred = control + delta).
- output_space == "all": wendet optional final_down_then_up an (analog zur Original-Pipeline).
- ReLU-Logik bleibt erhalten.
- NB / gene_decoder bleibt nutzbar.
"""

def __init__(self, *args, **kwargs):
    # Delta-spezifische kwargs herausziehen (damit sie nicht in PerturbationModel landen)
    self.use_delta_residual = kwargs.pop("use_delta_residual", True)
    self.delta_mode = kwargs.pop("delta_mode", "add")                 # add | gated
    self.delta_activation = kwargs.pop("delta_activation", "identity")  # identity | tanh | softsign
    self.delta_scale_mode = kwargs.pop("delta_scale", "none")         # none | auto | constant
    self.constant_delta_scale = kwargs.pop("constant_delta_scale", 1.0)
    self.beta_delta = kwargs.pop("beta_delta", 0.5)
    self.gamma_delta_sparsity = kwargs.pop("gamma_delta_sparsity", 0.0)
    self.clip_delta_norm = kwargs.pop("clip_delta_norm", None)
    self.log_delta_metrics = kwargs.pop("log_delta_metrics", True)
    self.gate_bias_init = kwargs.pop("gate_bias_init", 0.0)
    self.compute_delta_in = kwargs.pop("compute_delta_in", "log")     # "log" (einfach target - control), "raw" (expm1/log1p Pfad)

    super().__init__(*args, **kwargs)

    if not self.use_delta_residual:
        raise ValueError("DeltaStateTransitionPerturbationModel sollte nur genutzt werden, wenn use_delta_residual=True ist.")

    # Hidden- und Output-Dimensionen aus Basisklasse
    hidden_dim = self.hidden_dim
    output_dim = self.output_dim

    # Delta Head
    self.delta_head = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim)
    )

    if self.delta_mode == "gated":
        self.delta_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        self.delta_gate_bias = nn.Parameter(torch.full((output_dim,), self.gate_bias_init))
    else:
        self.delta_gate = None

    if self.delta_scale_mode == "auto":
        self.register_buffer("delta_running_std", torch.tensor(1.0))
        self.delta_momentum = 0.01

    # interner Cache für Forward (für Loss-Auswertung)
    self._delta_cache = None

# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------
def _activate_delta(self, x: torch.Tensor) -> torch.Tensor:
    if self.delta_activation == "identity":
        return x
    if self.delta_activation == "tanh":
        return torch.tanh(x)
    if self.delta_activation == "softsign":
        return x / (1 + torch.abs(x))
    raise ValueError(f"Unbekannte delta_activation={self.delta_activation}")

def _compute_delta_target(self, target: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    # Beide liegen (in deinem Setup) bereits im gleichen Raum (Embedding oder log1p Genes)
    if self.compute_delta_in == "log":
        return target - control
    elif self.compute_delta_in == "raw":
        # Falls es tatsächlich log1p-counts wären:
        target_raw = torch.expm1(target)
        control_raw = torch.expm1(control)
        delta_raw = target_raw - control_raw
        return torch.log1p(delta_raw.clamp(min=0))
    else:
        raise ValueError(f"compute_delta_in={self.compute_delta_in} nicht unterstützt")

def _append_confidence_token_if_needed(self, seq_input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    confidence_pred = None
    if self.confidence_token is not None:
        seq_input = self.confidence_token.append_confidence_token(seq_input)
    return seq_input, confidence_pred

# -------------------------------------------------
# Forward
# -------------------------------------------------
def forward(self, batch: Dict[str, torch.Tensor], padded: bool = True):
    """
    Repliziert den Aufbau des Originals:
    1. Reshape von pert_emb / ctrl_cell_emb
    2. Encoden (pert_encoder / basal_encoder)
    3. (Optional) batch_encoder addieren
    4. (Optional) confidence token anhängen
    5. Transformer
    6. confidence token extrahieren (falls aktiv)
    7. Delta-Head (statt project_out residual Logik)
    8. ReLU / final_down_then_up analog Basismodell

    Return:
      - output (flattened: [B*S, output_dim]) oder (output, confidence_pred)
    """
    if padded:
        pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
        basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
    else:
        pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
        basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

    pert_embedding = self.encode_perturbation(pert)      # [B,S,H]
    control_cells_enc = self.encode_basal_expression(basal)  # [B,S,H]

    seq_input = pert_embedding + control_cells_enc       # [B,S,H]

    # Batch-Encoder
    if self.batch_encoder is not None:
        batch_indices = batch["batch"]
        if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
            batch_indices = batch_indices.argmax(-1)
        if padded:
            batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
        else:
            batch_indices = batch_indices.reshape(1, -1)
        batch_emb = self.batch_encoder(batch_indices.long())  # [B,S,H]
        seq_input = seq_input + batch_emb

    # Confidence Token
    seq_input, _ = self._append_confidence_token_if_needed(seq_input)

    # Transformer + optional Mask
    if self.hparams.get("mask_attn", False):
        batch_size, seq_length, _ = seq_input.shape
        device = seq_input.device
        self.transformer_backbone._attn_implementation = "eager"
        base = torch.eye(seq_length, device=device).view(1, seq_length, seq_length)
        attn_mask = base.repeat(batch_size, 1, 1)
        outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
        transformer_output = outputs.last_hidden_state
    else:
        transformer_output = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state  # [B, S(+1), H] ggf.

    # Confidence Prediction extrahieren
    if self.confidence_token is not None:
        res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
    else:
        res_pred = transformer_output
        confidence_pred = None

    # Delta-Pfad
    # control_expr (Roh) direkt aus Batch (gleiche Dimension wie target)
    if padded:
        control_expr = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.output_dim)
        target_expr = batch["pert_cell_emb"].reshape(-1, self.cell_sentence_len, self.output_dim)
    else:
        control_expr = batch["ctrl_cell_emb"].reshape(1, -1, self.output_dim)
        target_expr = batch["pert_cell_emb"].reshape(1, -1, self.output_dim)

    # res_pred hat shape [B,S,H], wir benötigen delta_hat [B,S,O]
    delta_hat = self.delta_head(res_pred)
    delta_hat = self._activate_delta(delta_hat)

    if self.delta_gate is not None:
        gate_logits = self.delta_gate(res_pred) + self.delta_gate_bias
        gate = torch.sigmoid(gate_logits)
        delta_hat = gate * delta_hat

    if self.delta_scale_mode == "auto" and self.training:
        with torch.no_grad():
            batch_std = delta_hat.std().clamp(min=1e-6)
            self.delta_running_std = (1 - self.delta_momentum) * self.delta_running_std + self.delta_momentum * batch_std
        delta_hat = delta_hat / (self.delta_running_std + 1e-6)
    elif self.delta_scale_mode == "constant":
        delta_hat = delta_hat * self.constant_delta_scale

    preds_seq = control_expr + delta_hat  # Residual-Summe

    # Optionaler Pfad (all space)
    if self.output_space == "all" and hasattr(self, "final_down_then_up"):
        preds_seq = self.final_down_then_up(preds_seq)

    # ReLU wie Original (Gene/HVG)
    is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
    if is_gene_space or self.gene_decoder is None:
        preds_seq = self.relu(preds_seq)

    output = preds_seq.reshape(-1, self.output_dim)

    # Cache für Loss
    self._delta_cache = {
        "delta_hat": delta_hat,        # [B,S,O]
        "control_expr": control_expr,  # [B,S,O]
        "target_expr": target_expr,    # [B,S,O]
    }

    if confidence_pred is not None:
        return output, confidence_pred
    else:
        return output

# -------------------------------------------------
# Training
# -------------------------------------------------
def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True):
    if self.confidence_token is not None:
        pred, confidence_pred = self.forward(batch, padded=padded)
    else:
        pred = self.forward(batch, padded=padded)
        confidence_pred = None

    # Reshape für Haupt-Loss
    if padded:
        pred_seq = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"].reshape(-1, self.cell_sentence_len, self.output_dim)
    else:
        pred_seq = pred.reshape(1, -1, self.output_dim)
        target = batch["pert_cell_emb"].reshape(1, -1, self.output_dim)

    main_loss = self.loss_fn(pred_seq, target).nanmean()
    self.log("train_loss", main_loss)
    total_loss = main_loss

    # Delta-spezifischer Loss
    if self._delta_cache is not None:
        delta_hat = self._delta_cache["delta_hat"]
        control_expr = self._delta_cache["control_expr"]
        delta_target = self._compute_delta_target(target, control_expr)

        # Optional Norm-Clipping (pro Batch über gesamten Delta-Vektor)
        if self.clip_delta_norm is not None:
            with torch.no_grad():
                norms = torch.norm(delta_hat.view(delta_hat.size(0), -1), dim=1)
                scale = torch.clamp(self.clip_delta_norm / (norms + 1e-6), max=1.0)
            delta_hat.mul_(scale.view(-1, 1, 1))

        mse_delta = F.mse_loss(delta_hat, delta_target)
        sparsity = torch.mean(torch.abs(delta_hat))
        total_loss = total_loss + self.beta_delta * mse_delta + self.gamma_delta_sparsity * sparsity

        if self.log_delta_metrics:
            self.log("train/mse_delta", mse_delta)
            self.log("train/delta_sparsity", sparsity)

    # Gene Decoder / NB Decoder
    if self.gene_decoder is not None and "pert_cell_counts" in batch:
        gene_targets = batch["pert_cell_counts"]
        if isinstance(self.gene_decoder, NBDecoder):
            mu, theta = self.gene_decoder(pred_seq)
            gene_targets = gene_targets.reshape_as(mu)
            decoder_loss = nb_nll(gene_targets, mu, theta)
        else:
            if padded:
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            else:
                gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())
            gene_pred = self.gene_decoder(pred_seq)
            decoder_loss = self.loss_fn(gene_pred, gene_targets).mean()
        self.log("decoder_loss", decoder_loss)
        total_loss = total_loss + self.decoder_loss_weight * decoder_loss

    # Confidence Loss
    if confidence_pred is not None:
        loss_target = total_loss.detach().clone().unsqueeze(0) * 10
        if confidence_pred.dim() == 2:
            loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
        else:
            loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
        confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
        self.log("train/confidence_loss", confidence_loss)
        self.log("train/actual_loss", loss_target.mean())
        confidence_weight = 0.1
        total_loss = total_loss + confidence_weight * confidence_loss

    return total_loss

# -------------------------------------------------
# Validation
# -------------------------------------------------
def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
    if self.confidence_token is not None:
        pred, confidence_pred = self.forward(batch, padded=True)
    else:
        pred = self.forward(batch, padded=True)
        confidence_pred = None

    pred_seq = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
    target = batch["pert_cell_emb"].reshape(-1, self.cell_sentence_len, self.output_dim)

    main_loss = self.loss_fn(pred_seq, target).mean()
    loss = main_loss
    self.log("val_loss", main_loss)

    if self._delta_cache is not None:
        delta_hat = self._delta_cache["delta_hat"]
        control_expr = self._delta_cache["control_expr"]
        delta_target = self._compute_delta_target(target, control_expr)
        mse_delta = F.mse_loss(delta_hat, delta_target)
        loss = loss + self.beta_delta * mse_delta
        if self.log_delta_metrics:
            self.log("val/mse_delta", mse_delta)

    if self.gene_decoder is not None and "pert_cell_counts" in batch:
        gene_targets = batch["pert_cell_counts"]
        if isinstance(self.gene_decoder, NBDecoder):
            mu, theta = self.gene_decoder(pred_seq)
            gene_targets = gene_targets.reshape_as(mu)
            decoder_loss = nb_nll(gene_targets, mu, theta)
        else:
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            gene_pred = self.gene_decoder(pred_seq).reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            decoder_loss = self.loss_fn(gene_pred, gene_targets).mean()
        self.log("val/decoder_loss", decoder_loss)
        loss = loss + self.decoder_loss_weight * decoder_loss

    if confidence_pred is not None:
        loss_target = loss.detach().clone() * 10
        if confidence_pred.dim() == 2:
            loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
        else:
            loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
        confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
        self.log("val/confidence_loss", confidence_loss)
        self.log("val/actual_loss", loss_target.mean())

    return {"loss": loss, "predictions": pred_seq}

# -------------------------------------------------
# Test
# -------------------------------------------------
def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
    if self.confidence_token is not None:
        pred, confidence_pred = self.forward(batch, padded=False)
    else:
        pred = self.forward(batch, padded=False)
        confidence_pred = None

    pred_seq = pred.reshape(1, -1, self.output_dim)
    target = batch["pert_cell_emb"].reshape(1, -1, self.output_dim)
    main_loss = self.loss_fn(pred_seq, target).mean()
    loss = main_loss
    self.log("test_loss", main_loss)

    if self._delta_cache is not None:
        delta_hat = self._delta_cache["delta_hat"]
        control_expr = self._delta_cache["control_expr"]
        target_seq = self._delta_cache["target_expr"]
        delta_target = self._compute_delta_target(target_seq, control_expr)
        mse_delta = F.mse_loss(delta_hat, delta_target)
        loss = loss + self.beta_delta * mse_delta
        if self.log_delta_metrics:
            self.log("test/mse_delta", mse_delta)

    if self.gene_decoder is not None and "pert_cell_counts" in batch:
        gene_targets = batch["pert_cell_counts"]
        if isinstance(self.gene_decoder, NBDecoder):
            mu, theta = self.gene_decoder(pred_seq)
            gene_targets = gene_targets.reshape_as(mu)
            decoder_loss = nb_nll(gene_targets, mu, theta)
        else:
            gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())
            gene_pred = self.gene_decoder(pred_seq).reshape(1, -1, self.gene_decoder.gene_dim())
            decoder_loss = self.loss_fn(gene_pred, gene_targets).mean()
        self.log("test/decoder_loss", decoder_loss)
        loss = loss + self.decoder_loss_weight * decoder_loss

    if confidence_pred is not None:
        loss_target = loss.detach().clone() * 10
        if confidence_pred.dim() == 2:
            loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
        else:
            loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
        confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
        self.log("test/confidence_loss", confidence_loss)

    return {"loss": loss}

# -------------------------------------------------
# Predict
# -------------------------------------------------
def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True, **kwargs):
    if self.confidence_token is None:
        latent_output = self.forward(batch, padded=padded)
        confidence_pred = None
    else:
        latent_output, confidence_pred = self.forward(batch, padded=padded)

    output_dict = {
        "preds": latent_output,
        "pert_cell_emb": batch.get("pert_cell_emb", None),
        "pert_cell_counts": batch.get("pert_cell_counts", None),
        "pert_name": batch.get("pert_name", None),
        "celltype_name": batch.get("cell_type", None),
        "batch": batch.get("batch", None),
        "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        "pert_cell_barcode": batch.get("pert_cell_barcode", None),
        "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
    }

    if confidence_pred is not None:
        output_dict["confidence_pred"] = confidence_pred

    if self.gene_decoder is not None:
        if isinstance(self.gene_decoder, NBDecoder):
            mu, _ = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = mu
        else:
            output_dict["pert_cell_counts_preds"] = self.gene_decoder(latent_output)

    if self._delta_cache is not None:
        output_dict["delta"] = self._delta_cache["delta_hat"].reshape(-1, self.output_dim)
        output_dict["control_expr"] = self._delta_cache["control_expr"].reshape(-1, self.output_dim)

    return output_dict
