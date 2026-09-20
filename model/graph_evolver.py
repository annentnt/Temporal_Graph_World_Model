import torch
import torch.nn as nn
import torch.nn.functional as F


class SnapshotMessagePassingLayer(nn.Module):
    """One relational message-passing layer over a graph snapshot."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.message_layer = nn.Linear(dim, dim, bias=False)
        self.loop_layer = nn.Linear(dim, dim, bias=False)
        self.isolated_loop_layer = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state, rel_emb_table, edges, num_entities):
        """Runs one message-passing step and returns the updated entity states."""
        if edges.numel() == 0:
            loop = self.isolated_loop_layer(node_state)
            return self.dropout(self.norm(F.gelu(loop)))

        edges = edges.to(node_state.device)
        src, rel, dst = edges[:, 0], edges[:, 1], edges[:, 2]
        composed = node_state[src] + rel_emb_table[rel]
        messages = self.message_layer(composed)

        aggregated = torch.zeros_like(node_state)
        aggregated.index_add_(0, dst, messages)
        counts = torch.zeros(num_entities, 1, device=node_state.device, dtype=node_state.dtype)
        counts.index_add_(
            0,
            dst,
            torch.ones(messages.size(0), 1, device=node_state.device, dtype=node_state.dtype),
        )
        has_incoming = counts.squeeze(-1) > 0
        aggregated = aggregated / counts.clamp_min(1.0)

        loop_active = self.loop_layer(node_state)
        loop_isolated = self.isolated_loop_layer(node_state)
        loop = torch.where(has_incoming.unsqueeze(-1), loop_active, loop_isolated)

        return self.dropout(self.norm(F.gelu(loop + aggregated)))


class MultiStateGate(nn.Module):
    """Softmax gate that composes a new state with the full history of previous states."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.score_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history_states, new_state, related_emb=None):
        """Composes the history states and the new state into one output state."""
        score_inputs = history_states if related_emb is None else [
            state + related_emb for state in history_states
        ]
        logits = [self.score_proj(state) for state in score_inputs]
        logits.append(torch.zeros_like(new_state))
        weights = torch.softmax(torch.stack(logits, dim=0), dim=0)
        candidates = torch.stack(history_states + [new_state], dim=0)
        composed = (weights * candidates).sum(dim=0)
        return self.norm(self.dropout(composed))


def compute_related_relation_embedding(snapshot_edges, rel_emb_table, num_entities):
    """Averages relation embeddings per source entity for one snapshot."""
    dim = rel_emb_table.size(-1)
    device = rel_emb_table.device
    dtype = rel_emb_table.dtype
    if snapshot_edges is None or snapshot_edges.numel() == 0:
        return torch.zeros(num_entities, dim, device=device, dtype=dtype)

    snapshot_edges = snapshot_edges.to(device)
    src, rel = snapshot_edges[:, 0], snapshot_edges[:, 1]
    summed = torch.zeros(num_entities, dim, device=device, dtype=dtype)
    summed.index_add_(0, src, rel_emb_table[rel])
    counts = torch.zeros(num_entities, 1, device=device, dtype=dtype)
    counts.index_add_(0, src, torch.ones(src.size(0), 1, device=device, dtype=dtype))
    return summed / counts.clamp_min(1.0)


class GraphWorldStateEvolver(nn.Module):
    """Evolves entity states across a window of historical snapshots."""

    def __init__(self, dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.dim = int(dim)
        self.layers = nn.ModuleList(
            SnapshotMessagePassingLayer(self.dim, dropout=dropout)
            for _ in range(int(num_layers))
        )
        self.gate = MultiStateGate(self.dim, dropout=dropout)

    def forward(
        self,
        base_entity_state,
        rel_emb_table,
        time_ordered_edges,
        num_entities,
        same_snapshot_edges=None,
    ):
        """Runs the evolver over the window and returns the state at each snapshot time."""
        related_emb = compute_related_relation_embedding(
            same_snapshot_edges, rel_emb_table, num_entities,
        )
        history_states = [base_entity_state]
        states_by_time = {}
        for snapshot_time, edges in time_ordered_edges:
            propagated = history_states[-1]
            for layer in self.layers:
                propagated = layer(propagated, rel_emb_table, edges, num_entities)
            composed = self.gate(history_states, propagated, related_emb=related_emb)
            history_states.append(composed)
            states_by_time[snapshot_time] = composed
        return states_by_time
