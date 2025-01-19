"""
This is a simplified version of the GCMol implementation from:
https://github.com/Shihang-Wang-58/PhenoScreen/blob/main/phenoscreen/model/gcmol.py

The core GNN-based molecular encoding functionality has been preserved, while removing
unnecessary components like image encoding and other auxiliary functions.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np

from dgl import batch
from dgllife.utils import (
    smiles_to_bigraph,
    BaseAtomFeaturizer,
    CanonicalBondFeaturizer,
    ConcatFeaturizer,
    atom_type_one_hot,
    atom_formal_charge,
    atom_hybridization_one_hot,
    atom_chiral_tag_one_hot,
    atom_is_in_ring,
    atom_is_aromatic,
)
from dgllife.model.gnn.wln import WLN
from dgllife.model.readout.mlp_readout import MLPNodeReadout


class MolecularEncoder(nn.Module):
    """
    A minimal GNN-based encoder that transforms a DGL molecular graph
    into a fixed-size embedding (optionally on CPU or GPU).
    """

    def __init__(
        self,
        atom_feat_size=None,
        bond_feat_size=None,
        num_features=512,
        num_layers=4,
        num_out_features=1024,
        activation="LeakyReLU",
        readout_type="MeanMLP",
        gnn_type="WLN",
        device="cpu",
    ):
        super().__init__()

        # Store device internally (as a torch.device object).
        # If user requested CUDA, but none is available, fall back to CPU.
        if device.lower() == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        # Basic activation dictionary
        activation_dict = {
            "GELU": nn.GELU(),
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Tanh": nn.Tanh(),
            "SiLU": nn.SiLU(),
            "ELU": nn.ELU(),
            "SELU": nn.SELU(),
            "CELU": nn.CELU(),
            "PReLU": nn.PReLU(),
        }

        # GNN backbone (WLN)
        self.GeminiEncoder = WLN(
            atom_feat_size,
            bond_feat_size,
            n_layers=num_layers,
            node_out_feats=num_features,
        )

        # Minimal readout: MeanMLP
        self.readout = MLPNodeReadout(
            node_feats=num_features,
            hidden_feats=num_features,
            graph_feats=num_features,
            activation=activation_dict[activation],
            mode="mean",
        )

        # Finally, move everything to CPU or GPU as requested.
        self.to(self.device)

    def forward(self, mol_graph):
        # WLN expects node and bond features
        encoding = self.GeminiEncoder(
            mol_graph,
            mol_graph.ndata["atom_type"],
            mol_graph.edata["bond_type"],
        )
        return self.readout(mol_graph, encoding)


class MCP_Matching(nn.Module):
    """
    A minimal base class that builds a MolecularEncoder and
    provides a `mol_encode` method for SMILES strings.
    """

    def __init__(
        self,
        model_name,
        batch_size=128,
        device="cpu",  # new
        # The rest are placeholders to align with GCMol usage
        feature_list=["smiles1", "smiles2"],
        label_dict=None,
        encoding_features=2048,
        metric_list=None,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size

        # If user requested CUDA but not available, fall back to CPU:
        if device.lower() == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        # DGL atom/bond featurizers
        self.atom_featurizer = BaseAtomFeaturizer(
            featurizer_funcs={
                "atom_type": ConcatFeaturizer(
                    [
                        atom_type_one_hot,
                        atom_hybridization_one_hot,
                        atom_formal_charge,
                        atom_chiral_tag_one_hot,
                        atom_is_in_ring,
                        atom_is_aromatic,
                    ]
                )
            }
        )
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="bond_type")

        # Build a MolecularEncoder
        self.Encoder = MolecularEncoder(
            atom_feat_size=self.atom_featurizer.feat_size("atom_type"),
            bond_feat_size=self.bond_featurizer.feat_size("bond_type"),
            num_layers=4,
            num_features=2048,  # can be changed via model_params.json
            activation="LeakyReLU",
            readout_type="MeanMLP",
            gnn_type="WLN",
            device=device,
        )

    def smiles2tensor(self, input_sents):
        # Turn a list of SMILES into a batched DGLGraph on CUDA
        graphs = []
        for smi in input_sents:
            g = smiles_to_bigraph(
                smi,
                node_featurizer=self.atom_featurizer,
                edge_featurizer=self.bond_featurizer,
            )
            if g is not None:  # Only process valid graphs
                g = g.to(self.device)
                graphs.append(g)
        if not graphs:  # Handle case where no valid graphs were created
            raise ValueError(
                "No valid molecular graphs could be created from input SMILES"
            )
        return batch(graphs)

    def mol_encode(self, input_sents):
        """
        Encode a list of SMILES into embeddings (on self.device),
        then return them on CPU by default (to avoid GPU memory usage).
        """
        self.eval()
        with torch.no_grad():
            # Convert SMILES to a batched DGL graph
            input_tensor = self.smiles2tensor(input_sents)
            # Forward pass
            features = self.Encoder(input_tensor)
            # Move embeddings back to CPU (so user sees a CPU tensor)
            return features.cpu()


class GCMol(MCP_Matching):
    """
    Loads GCMol from:
        <model_name>/model_params.json
    and
        <model_name>/GCMol.pt

    Then provides `mol_encode(smiles_list)` for embeddings.
    """

    def __init__(
        self,
        model_name,
        batch_size=None,
        similarity_metrics_list=["Cosine", "Pearson", "RMSE", "Manhattan"],
        device="cpu",
    ):
        # Load JSON hyperparameters
        with open(f"{model_name}/model_params.json", "r", encoding="utf-8") as f:
            self.params = json.load(f)

        # If user provides batch_size, override
        if batch_size is not None:
            self.params["batch_size"] = batch_size

        # Insert device into params so MCP_Matching sees it
        self.params["device"] = device

        # Initialize base class
        super().__init__(model_name, **self.params)
        self.similarity_metrics_list = similarity_metrics_list

        ckpt_path = os.path.join(model_name, "GCMol.pt")
        if os.path.isfile(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            # print("Missing keys in new model:", missing)
            # print("Unused keys in checkpoint:", unexpected)
        else:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        self.eval()
