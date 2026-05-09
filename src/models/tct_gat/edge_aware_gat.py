from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class EdgeAwareRelationAttention(nn.Module):
    """Sparse top-K relation attention for one directed relation."""

    def __init__(
        self,
        token_dim: int,
        edge_dim: int,
        heads: int,
        edge_embedding_dim: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if token_dim % heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by heads={heads}.")
        self.token_dim = int(token_dim)
        self.heads = int(heads)
        self.head_dim = int(token_dim // heads)
        self.src_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.dst_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, heads * edge_embedding_dim, bias=False)
        self.edge_to_value = nn.Linear(edge_dim, token_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, 2 * self.head_dim + edge_embedding_dim))
        self.dropout = nn.Dropout(attention_dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attn)

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        neighbor_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if source_tokens.ndim != 3 or target_tokens.ndim != 3:
            raise ValueError("source_tokens and target_tokens must have shape B x S x D.")
        batch_size, num_stations, _ = target_tokens.shape
        k_neighbors = int(neighbor_index.shape[1])

        src_projected = self.src_proj(source_tokens)
        dst_projected = self.dst_proj(target_tokens).view(batch_size, num_stations, self.heads, self.head_dim)
        flat_neighbors = neighbor_index.reshape(-1)
        gathered_src = src_projected.index_select(1, flat_neighbors).view(
            batch_size,
            num_stations,
            k_neighbors,
            self.heads,
            self.head_dim,
        )
        dst = dst_projected.unsqueeze(2).expand(-1, -1, k_neighbors, -1, -1)
        edge_emb = self.edge_proj(edge_attr).view(num_stations, k_neighbors, self.heads, -1)
        edge_emb = edge_emb.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        attn_input = torch.cat([dst, gathered_src, edge_emb], dim=-1)
        scores = self.leaky_relu((attn_input * self.attn.view(1, 1, 1, self.heads, -1)).sum(dim=-1))
        alpha = torch.softmax(scores, dim=2)
        alpha = self.dropout(alpha)

        edge_value = self.edge_to_value(edge_attr).view(num_stations, k_neighbors, self.heads, self.head_dim)
        edge_value = edge_value.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        message = (alpha.unsqueeze(-1) * (gathered_src + edge_value)).sum(dim=2)
        return message.reshape(batch_size, num_stations, self.token_dim)


class EdgeAwareHeteroGATLayer(nn.Module):
    """One hetero GAT layer over rental and return station tokens."""

    def __init__(
        self,
        token_dim: int,
        edge_dim: int,
        heads: int = 4,
        edge_embedding_dim: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rr = EdgeAwareRelationAttention(token_dim, edge_dim, heads, edge_embedding_dim, attention_dropout)
        self.dd = EdgeAwareRelationAttention(token_dim, edge_dim, heads, edge_embedding_dim, attention_dropout)
        self.rd = EdgeAwareRelationAttention(token_dim, edge_dim, heads, edge_embedding_dim, attention_dropout)
        self.dr = EdgeAwareRelationAttention(token_dim, edge_dim, heads, edge_embedding_dim, attention_dropout)
        self.rental_mix = nn.Linear(2 * token_dim, token_dim)
        self.return_mix = nn.Linear(2 * token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        self.rental_norm1 = nn.LayerNorm(token_dim)
        self.return_norm1 = nn.LayerNorm(token_dim)
        self.rental_norm2 = nn.LayerNorm(token_dim)
        self.return_norm2 = nn.LayerNorm(token_dim)
        self.rental_ffn = nn.Sequential(nn.Linear(token_dim, 4 * token_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * token_dim, token_dim))
        self.return_ffn = nn.Sequential(nn.Linear(token_dim, 4 * token_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * token_dim, token_dim))

    def forward(
        self,
        rental_tokens: torch.Tensor,
        return_tokens: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rr_msg = self.rr(rental_tokens, rental_tokens, graph["neighbor_index_rr"], graph["edge_attr_rr"])
        dr_msg = self.dr(return_tokens, rental_tokens, graph["neighbor_index_dr"], graph["edge_attr_dr"])
        dd_msg = self.dd(return_tokens, return_tokens, graph["neighbor_index_dd"], graph["edge_attr_dd"])
        rd_msg = self.rd(rental_tokens, return_tokens, graph["neighbor_index_rd"], graph["edge_attr_rd"])
        rental_message = self.rental_mix(torch.cat([rr_msg, dr_msg], dim=-1))
        return_message = self.return_mix(torch.cat([dd_msg, rd_msg], dim=-1))
        rental_tokens = self.rental_norm1(rental_tokens + self.dropout(rental_message))
        return_tokens = self.return_norm1(return_tokens + self.dropout(return_message))
        rental_tokens = self.rental_norm2(rental_tokens + self.dropout(self.rental_ffn(rental_tokens)))
        return_tokens = self.return_norm2(return_tokens + self.dropout(self.return_ffn(return_tokens)))
        return rental_tokens, return_tokens


class EdgeAwareHeteroGAT(nn.Module):
    def __init__(
        self,
        graph_dir: str | Path,
        token_dim: int = 128,
        edge_dim: int | None = None,
        layers: int = 1,
        heads: int = 4,
        edge_embedding_dim: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        graph_path = Path(graph_dir)
        graph_arrays = {
            "neighbor_index_rr": np.load(graph_path / "neighbor_index_rr.npy"),
            "neighbor_index_dd": np.load(graph_path / "neighbor_index_dd.npy"),
            "neighbor_index_rd": np.load(graph_path / "neighbor_index_rd.npy"),
            "neighbor_index_dr": np.load(graph_path / "neighbor_index_dr.npy"),
            "edge_attr_rr": np.load(graph_path / "edge_attr_rr.npy"),
            "edge_attr_dd": np.load(graph_path / "edge_attr_dd.npy"),
            "edge_attr_rd": np.load(graph_path / "edge_attr_rd.npy"),
            "edge_attr_dr": np.load(graph_path / "edge_attr_dr.npy"),
        }
        inferred_edge_dim = int(graph_arrays["edge_attr_rr"].shape[-1])
        if edge_dim is not None and int(edge_dim) != inferred_edge_dim:
            raise ValueError(f"Configured edge_dim={edge_dim} differs from graph edge dim={inferred_edge_dim}.")
        for name, array in graph_arrays.items():
            dtype = torch.long if name.startswith("neighbor_index") else torch.float32
            self.register_buffer(name, torch.as_tensor(array, dtype=dtype), persistent=True)
        self.layers = nn.ModuleList(
            [
                EdgeAwareHeteroGATLayer(
                    token_dim=token_dim,
                    edge_dim=inferred_edge_dim,
                    heads=heads,
                    edge_embedding_dim=edge_embedding_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(layers)
            ]
        )

    @property
    def graph_buffers(self) -> dict[str, torch.Tensor]:
        return {
            "neighbor_index_rr": self.neighbor_index_rr,
            "neighbor_index_dd": self.neighbor_index_dd,
            "neighbor_index_rd": self.neighbor_index_rd,
            "neighbor_index_dr": self.neighbor_index_dr,
            "edge_attr_rr": self.edge_attr_rr,
            "edge_attr_dd": self.edge_attr_dd,
            "edge_attr_rd": self.edge_attr_rd,
            "edge_attr_dr": self.edge_attr_dr,
        }

    def forward(self, rental_tokens: torch.Tensor, return_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.graph_buffers
        for layer in self.layers:
            rental_tokens, return_tokens = layer(rental_tokens, return_tokens, graph)
        return rental_tokens, return_tokens
