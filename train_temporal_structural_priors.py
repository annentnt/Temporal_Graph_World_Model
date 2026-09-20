import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.seed import seed_everything


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_quadruples(processed_dir):
    path = os.path.join(processed_dir, 'train_quadruples.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run utils/preprocess_temporal_data.py first."
        )
    quads = torch.load(path, map_location='cpu').long()
    if quads.dim() != 2 or quads.size(1) != 4:
        raise ValueError(f"Expected train_quadruples.pt with shape (N, 4), got {tuple(quads.shape)}")
    return quads


def group_edges_by_time(quadruples):
    snapshots = {}
    for time_id in torch.unique(quadruples[:, 3], sorted=True).tolist():
        mask = quadruples[:, 3] == int(time_id)
        snapshots[int(time_id)] = quadruples[mask][:, :3].contiguous()
    return snapshots


class SnapshotCompGCNPrior(nn.Module):
    """CompGCN-style relational message-passing model for one graph snapshot."""

    def __init__(self, num_entities, num_relations, dim, layers=1, dropout=0.1):
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.dim = int(dim)
        self.entity_emb = nn.Embedding(self.num_entities, self.dim)
        self.relation_emb = nn.Embedding(self.num_relations, self.dim)
        self.message_layers = nn.ModuleList(
            nn.Linear(self.dim, self.dim, bias=False) for _ in range(int(layers))
        )
        self.loop_layers = nn.ModuleList(
            nn.Linear(self.dim, self.dim, bias=False) for _ in range(int(layers))
        )
        self.norm_layers = nn.ModuleList(
            nn.LayerNorm(self.dim) for _ in range(int(layers))
        )
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        for layer in list(self.message_layers) + list(self.loop_layers):
            nn.init.xavier_uniform_(layer.weight)

    def encode_snapshot(self, edges):
        src = edges[:, 0]
        rel = edges[:, 1]
        dst = edges[:, 2]

        node_state = self.entity_emb.weight
        for message_layer, loop_layer, norm_layer in zip(
            self.message_layers,
            self.loop_layers,
            self.norm_layers,
        ):
            composed = node_state[src] + self.relation_emb(rel)
            messages = message_layer(composed)

            aggregated = torch.zeros_like(node_state)
            aggregated.index_add_(0, dst, messages)
            counts = torch.zeros(
                self.num_entities,
                1,
                device=node_state.device,
                dtype=node_state.dtype,
            )
            counts.index_add_(
                0,
                dst,
                torch.ones(messages.size(0), 1, device=node_state.device, dtype=node_state.dtype),
            )
            aggregated = aggregated / counts.clamp_min(1.0)

            loop = loop_layer(node_state)
            node_state = norm_layer(F.gelu(loop + aggregated))
            node_state = self.dropout(node_state)

        return F.normalize(node_state, p=2, dim=-1)

    def score_batch(self, snapshot_node_state, triples):
        h = triples[:, 0]
        r = triples[:, 1]
        query = snapshot_node_state[h] + self.relation_emb(r)
        query = F.normalize(query, p=2, dim=-1)
        return torch.mm(query, snapshot_node_state.t())


def iter_snapshot_batches(edges, batch_size, shuffle=True):
    indices = list(range(edges.size(0)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield edges[batch_idx]


def train(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    entity2id = load_json(os.path.join(args.processed_dir, 'entity2id.json'))
    relation2id = load_json(os.path.join(args.processed_dir, 'relation2id.json'))
    train_quads = load_quadruples(args.processed_dir)
    snapshots = group_edges_by_time(train_quads)
    snapshot_ids = sorted(snapshots)

    model = SnapshotCompGCNPrior(
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        dim=args.dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Using device: {device}")
    print(f"Training CompGCN structural priors on {len(snapshot_ids)} snapshots")
    print(f"Entities={len(entity2id)}, Relations={len(relation2id)}, dim={args.dim}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0
        random.shuffle(snapshot_ids)

        pbar = tqdm(snapshot_ids, desc=f"Epoch {epoch + 1} [CompGCN]")
        for time_id in pbar:
            edges = snapshots[time_id].to(device)
            if edges.numel() == 0:
                continue

            for batch_edges in iter_snapshot_batches(edges, args.batch_size, shuffle=True):
                snapshot_state = model.encode_snapshot(edges)
                scores = model.score_batch(snapshot_state, batch_edges) / args.temperature
                target = batch_edges[:, 2]
                loss = F.cross_entropy(scores, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1
                pbar.set_postfix(loss=total_loss / max(total_batches, 1))

        avg_loss = total_loss / max(total_batches, 1)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

    model.eval()
    entity_accumulator = torch.zeros(
        len(entity2id),
        args.dim,
        device=device,
    )
    entity_counts = torch.zeros(
        len(entity2id),
        1,
        device=device,
    )

    with torch.no_grad():
        for time_id in tqdm(sorted(snapshots), desc="Averaging snapshot states"):
            edges = snapshots[time_id].to(device)
            snapshot_state = model.encode_snapshot(edges)
            touched = torch.unique(torch.cat([edges[:, 0], edges[:, 2]], dim=0))
            entity_accumulator[touched] += snapshot_state[touched]
            entity_counts[touched] += 1.0

        fallback = model.entity_emb.weight
        structural_entities = torch.where(
            entity_counts > 0,
            entity_accumulator / entity_counts.clamp_min(1.0),
            fallback,
        )
        structural_entities = F.normalize(structural_entities, p=2, dim=-1).detach().cpu()
        structural_relations = F.normalize(model.relation_emb.weight, p=2, dim=-1).detach().cpu()

    os.makedirs(args.processed_dir, exist_ok=True)
    entity_path = os.path.join(args.processed_dir, 'structural_entities.pt')
    relation_path = os.path.join(args.processed_dir, 'structural_relations.pt')
    torch.save(structural_entities.contiguous(), entity_path)
    torch.save(structural_relations.contiguous(), relation_path)
    print(f"Saved CompGCN structural entities to: {entity_path}")
    print(f"Saved CompGCN structural relations to: {relation_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal structural priors with snapshot CompGCN."
    )
    parser.add_argument('--processed_dir', type=str, default='data-processed/icews05-15')
    parser.add_argument('--dim', type=int, default=200)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
