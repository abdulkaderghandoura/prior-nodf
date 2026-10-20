# Adapted from NODF-HashEnc (https://github.com/MunzerDw/nodf-hashenc), which builds on
# NODF (https://github.com/Will-Consagra/NODF); used with the authors' permission.
import argparse
from typing import Dict

import torch
from torch import optim
import pytorch_lightning as pl

from models.inr import INR
from models.hash_embeddings import HashEmbedder
from utils.utility import get_phi_r_tensors


class NODF(pl.LightningModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # Required only for training to calculate the penalized maximum likelihood estimate
        Phi_tensor, R_tensor = get_phi_r_tensors(args)
        self.register_buffer("R_tensor", R_tensor, persistent=False)
        self.register_buffer("Phi_tensor", Phi_tensor, persistent=False)

        # Get the number of coefficients
        K = int((args.sh_order + 1) * (args.sh_order + 2) / 2)

        # Initialize the INR
        if args.use_baseline:
            self.inr = INR(
                in_features=3,
                out_features=K,
                hidden_features=args.r,
                hidden_layers=args.depth,
                first_omega_0=args.omega0,
                hidden_omega_0=args.omega0_hidden,
                sigma0=args.sigma0,
                inr=self.args.inr,
                skip_conn=self.args.skip_conn,
                batchnorm=self.args.batchnorm,
            )

        else:
            self.inr = INR(
                in_features=(self.args.n_levels * self.args.n_features_per_level) + 3,
                out_features=K,
                hidden_features=args.r,
                hidden_layers=args.depth,
                first_omega_0=args.omega0,
                hidden_omega_0=args.omega0_hidden,
                sigma0=args.sigma0,
                inr=self.args.inr,
                skip_conn=self.args.skip_conn,
                batchnorm=self.args.batchnorm,
            )
            self.hash_embedder = HashEmbedder(
                n_levels=self.args.n_levels,
                n_features_per_level=self.args.n_features_per_level,
                log2_hashmap_size=self.args.log2_hashmap_size,
                base_resolution=self.args.base_resolution,
                per_level_scale=self.args.per_level_scale,
            )

        if self.args.freeze_mlp_except_last:
            self._freeze_mlp_layers(retain_last=self.args.freeze_mlp_except_last)

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        batch: dict containing "coords" (B, 3) or (B * (M + 1), 3)

        returns: torch.Tensor (N, K) of SH coefficients, one row per coordinate
        """
        coords = batch["coords"]

        inr = self.inr

        if self.args.use_baseline:
            model_output = inr(coords)

        else:
            embeddings = self.hash_embedder(coords)
            model_input = torch.cat([embeddings, coords], dim=-1)
            model_output = inr(model_input)

        return model_output

    def training_step(self, batch, batch_idx):
        """
        'training_step' is the method that is called for every batch of data in the training loop.

        batch: dict containing "coords" (B, 3), "aux_coords" (B, N, 3), and "signal" (B, M)
        batch_idx: int

        returns: dict of torch.tensor losses
        """
        if self.args.downsampling_subgrid_size is not None:
            B, N = batch["aux_coords"].shape[:2]

            # Number of downsampled points to be used
            M = min(self.args.num_downsampling_points, N)

            # Generate random indices for each row
            weights = torch.ones(B, N, device=batch["aux_coords"].device)  # Uniform weights
            indices = torch.multinomial(weights, M, replacement=False)  # (B, M)

            # Select M random auxiliary points per row
            selected_aux_coords = batch["aux_coords"][torch.arange(B).unsqueeze(1), indices]  # (B, M, 3)

            # Combine main and selected auxiliary points
            combined_coords = torch.cat([batch["coords"].unsqueeze(1), selected_aux_coords], dim=1)  # (B, M + 1, 3)

            # Run forward on the combined coordinates
            combined_coords_dict = {"coords": combined_coords.view(B * (M + 1), 3)}
            combined_chat = self.forward(combined_coords_dict)  # (B * (M + 1), K)

            # Reshape and average the model outputs, including the main coordinates
            combined_chat = combined_chat.view(B, M + 1, -1)  # (B, M + 1, K)
            chat = torch.mean(combined_chat, dim=1)  # (B, K)
        else:
            chat = self.forward(batch)  # (B, K)

        # For total variation prior
        chat_difference_vectors = None

        if self.args.tv_directions:
            # The low-resolution data are 4x coarser along z (through-plane downsampling), so the
            # z step is 4x larger: each difference spans about the same number of voxels per axis
            scaled_offset = self.args.offset_tv * torch.ones(3, device=batch['coords'].device)
            scaled_offset[2] = self.args.offset_tv * 4  # the through-plane downsampling factor

            chat_difference_vectors = []
            for i, axis in enumerate("xyz"):
                if axis in self.args.tv_directions:
                    coords_positive = batch['coords'].clone()
                    coords_negative = batch['coords'].clone()

                    coords_positive[:, i] = torch.clamp(coords_positive[:, i] + scaled_offset[i], 0, 1)
                    coords_negative[:, i] = torch.clamp(coords_negative[:, i] - scaled_offset[i], 0, 1)

                    coords_both = torch.cat([coords_positive, coords_negative], dim=0)  # (2B, 3)
                    chat_both = self.forward({"coords": coords_both})  # (2B, K)
                    chat_positive, chat_negative = chat_both.split(coords_positive.shape[0], dim=0)

                    # Add chat difference
                    chat_difference_vectors.append(chat_positive - chat_negative)  # B x K

            if chat_difference_vectors:
                chat_difference_vectors = torch.stack(chat_difference_vectors, dim=-1)  # (B, K, 3)

        losses = self._neg_pmle(chat, chat_difference_vectors, batch["signal"], batch['coords'])

        # Compute weighted total loss
        loss_weights = {
            "prior_energy": self.args.lambda_c,
            "l2_loss": 1.0,
            "total_variation": self.args.lambda_tv,
        }
        total_loss = sum(loss_weights[k] * v for k, v in losses.items())
        losses["loss"] = total_loss

        # Logging
        for key, value in losses.items():
            self.log(
                f"{key}/train",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=False,
                sync_dist=True,
                batch_size=batch["coords"].shape[0]
            )

        return {"loss": losses["loss"]}

    def configure_optimizers(self):
        """
        configure the optimizer and scheduler according to Pytorch Lightning
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        return optimizer

    def _neg_pmle(
        self,
        chat: torch.Tensor,
        chat_difference_vectors: torch.Tensor,
        signal: torch.Tensor,
        coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        chat: torch.tensor (N x K) of coefficient field evaluations
        chat_difference_vectors: torch.Tensor (B x K x 3) difference of coefficient field evaluations w.r.t. neighborhood
        signal: torch.tensor (N x M) of (noisy + discretized) function samples
        coords: torch.tensor (N x 3) of (noisy + discretized) input coordinates

        returns: dict of torch.tensor losses
        """
        l2_loss = self._neg_log_likelihood(chat, signal, self.Phi_tensor)
        prior_energy = self._integrated_roughness(chat, self.R_tensor)
        total_variation = torch.tensor(0.0, device=chat.device)
        if chat_difference_vectors is not None:
            total_variation = self._total_variation(chat_difference_vectors, coords, self.Phi_tensor)

        return {
            "l2_loss": l2_loss,
            "prior_energy": prior_energy,
            "total_variation": total_variation
        }

    def _neg_log_likelihood(
        self,
        chat: torch.Tensor,
        signal: torch.Tensor,
        Phi_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        chat: B X K of coefficient field evaluations
        signal: B X M of (noisy + discretized) function samples
        Phi_tensor: torch.tensor (M x K) basis evaluation matrix

        returns: torch.tensor batch mean of negative log likelihood
        """
        yhat = chat @ Phi_tensor.T  # B x M (flattened predicted tensor)
        l2_loss = ((yhat - signal) ** 2).mean()
        return l2_loss

    def _integrated_roughness(
        self,
        chat: torch.Tensor,
        R_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        chat: B X K of coefficient field evaluations
        R_tensor: K x K (diagonal) prior precision matrix of functions

        returns: torch.tensor batch mean of integrated roughness
        """
        energy = torch.mean((chat**2) * torch.diag(R_tensor))
        return energy

    def _total_variation(
        self,
        chat_difference_vectors: torch.Tensor,
        coords: torch.Tensor,
        Phi_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Total-variation penalty: the mean L2 norm of the finite differences of the
        coefficient field along the selected axes.

        chat_difference_vectors: torch.Tensor (B x K x D) differences along D axes
        coords, Phi_tensor: unused

        returns: torch.Tensor scalar penalty
        """
        tv_loss = torch.norm(chat_difference_vectors, p=2, dim=-1).mean()

        return tv_loss

    def _freeze_mlp_layers(self, retain_last=1):
        """Freeze all MLP layers except the last `retain_last` layers."""
        num_layers = len(self.inr.net)
        assert retain_last <= num_layers

        for layer in self.inr.net[:-retain_last]:
            for param in layer.parameters():
                param.requires_grad = False

        # Unfreeze the last `retain_last` layers
        for layer in self.inr.net[-retain_last:]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"Freezing all layers except the last {retain_last} layers.")

    def count_parameters(self):
        """
        Counts and returns the number of trainable parameters in the model.
        """
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            print(f"{name}: {params}")
            total_params += params
        print(f"Total Trainable Params: {total_params}")
        return total_params
