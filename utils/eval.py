import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EntityDataset(Dataset):
    """Dataset that yields every entity id, used to encode all candidates."""
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'entity2id.json'), 'r') as f:
            self.entity2id = json.load(f)

        self.num_entities = len(self.entity2id)

    def __len__(self):
        return self.num_entities

    def __getitem__(self, idx):
        return {
            'id': idx,
        }


def load_triples_for_filtering(data_dir, splits=None):
    """Loads all triples/quadruples from the given splits into a set."""
    if splits is None:
        splits = ['train']

    all_triples = set()
    for split in splits:
        path = os.path.join(data_dir, f'{split}_quadruples.pt')
        if not os.path.exists(path):
            path = os.path.join(data_dir, f'{split}_triples.pt')
        if os.path.exists(path):
            triples = torch.load(path)
            for row in triples:
                if row.numel() == 4:
                    h, r, t, time_id = row
                    all_triples.add((h.item(), r.item(), t.item(), time_id.item()))
                else:
                    h, r, t = row
                    all_triples.add((h.item(), r.item(), t.item()))
    return all_triples


def load_hr_map_for_filtering(data_dir, preferred_ground_truth_file=None, fallback_splits=None):
    """Builds a (head, relation[, time]) -> true tails map for filtered ranking."""
    if fallback_splits is None:
        fallback_splits = ['train']

    if preferred_ground_truth_file is not None:
        gt_path = os.path.join(data_dir, preferred_ground_truth_file)
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_json = json.load(f)

            hr_map = {}
            for key, tails in gt_json.items():
                parts = tuple(map(int, key.split(',')))
                hr_map[parts] = set(int(t) for t in tails)
            return hr_map

    all_triples = load_triples_for_filtering(data_dir, splits=fallback_splits)
    hr_map = {}
    for row in all_triples:
        if len(row) == 4:
            h, r, t, time_id = row
            key = (h, r, time_id)
        else:
            h, r, t = row
            key = (h, r)
        if key not in hr_map:
            hr_map[key] = set()
        hr_map[key].add(t)
    return hr_map


def build_entity_loader(data_dir, batch_size, num_workers=2):
    """Builds a DataLoader that iterates over every entity id."""
    entity_dataset = EntityDataset(data_dir)

    def entity_collate(batch):
        ids = [x['id'] for x in batch]
        return {
            'id': torch.tensor(ids)
        }

    return DataLoader(
        entity_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=entity_collate,
        num_workers=num_workers
    )


def encode_all_entities_as_targets(model, entity_loader, device):
    """Encodes every entity once, for the non-temporal static candidate table."""
    all_chunks = []
    model.eval()
    with torch.no_grad():
        for batch in entity_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            all_chunks.append(model.encode_target(batch).cpu())
    return torch.cat(all_chunks, dim=0).to(device)


def compute_filtered_ranking_metrics(
    model,
    data_loader,
    all_entity_embeddings,
    hr_map,
    device,
    desc="Filtered Ranking",
    save_predictions_path=None,
    topk=50,
    edge_bank=None,
    history_len=None,
):
    """Runs filtered-ranking evaluation over data_loader and returns MRR/MR/Hits@k."""
    hits1, hits3, hits10, mrr, mr = 0, 0, 0, 0.0, 0.0
    total = 0

    writer = None
    if save_predictions_path is not None:
        os.makedirs(os.path.dirname(save_predictions_path), exist_ok=True)
        writer = open(save_predictions_path, 'w', encoding='utf-8')

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

            t_ids = batch['t_batch']['id'].to(device)
            time_ids = (
                batch['t_batch']['time_id'].to(device)
                if batch['t_batch'].get('time_id') is not None
                else None
            )
            h_ids = batch['h_batch']['id'].cpu().numpy()
            r_ids = batch['r_batch']['id'].cpu().numpy()

            graph_state_by_time = None
            if edge_bank is not None and h_batch.get('time_id') is not None:
                required_time_ids = [h_batch['time_id']]
                ctx_time_ids = context_batch.get('time_id')
                if ctx_time_ids is not None and ctx_time_ids.numel() > 0:
                    required_time_ids.append(ctx_time_ids)
                graph_state_by_time = model.build_graph_world_states(
                    time_ids=torch.cat(required_time_ids),
                    edge_bank=edge_bank,
                    history_len=history_len,
                )

            if getattr(model, 'decoder_name', 'dot') == 'convtranse':
                scores = model.score_all_entities(
                    h_batch, r_batch, context_batch, graph_state_by_time=graph_state_by_time,
                )
            elif batch['t_batch'].get('time_id') is None:
                query_vectors = model(h_batch, r_batch, context_batch)
                scores = torch.mm(query_vectors, all_entity_embeddings.t())
                scores = scores / model.temperature
            else:
                query_vectors, relation_vectors = model.encode_query(
                    h_batch, r_batch, context_batch, graph_state_by_time=graph_state_by_time,
                )
                query_time_ids = batch['t_batch']['time_id'].to(device)
                if graph_state_by_time is not None:
                    scores = model._score_with_per_time_candidates(
                        query_vectors, relation_vectors, query_time_ids, graph_state_by_time,
                    )
                else:
                    num_entities = all_entity_embeddings.size(0) if all_entity_embeddings is not None else model.config.num_entities
                    entity_ids = torch.arange(num_entities, device=device, dtype=torch.long)
                    scores = torch.empty(
                        query_vectors.size(0),
                        num_entities,
                        device=device,
                        dtype=query_vectors.dtype,
                    )
                    for time_id in torch.unique(query_time_ids, sorted=True):
                        row_mask = query_time_ids == time_id
                        candidate_batch = {
                            'id': entity_ids,
                            'time_id': torch.full_like(entity_ids, int(time_id.item())),
                        }
                        candidate_embeddings = model.encode_target(candidate_batch)
                        scores[row_mask] = torch.mm(
                            query_vectors[row_mask],
                            candidate_embeddings.t(),
                        ) / model.temperature

            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

            for i in range(scores.size(0)):
                h_id = h_ids[i]
                r_id = r_ids[i]
                true_t = t_ids[i].item()
                if time_ids is not None:
                    filter_key = (h_id, r_id, int(time_ids[i].item()))
                else:
                    filter_key = (h_id, r_id)

                filter_mask_indices = list(hr_map.get(filter_key, []))
                if true_t in filter_mask_indices:
                    filter_mask_indices.remove(true_t)

                if filter_mask_indices:
                    scores[i, filter_mask_indices] = -float('inf')

            target_scores = scores.gather(1, t_ids.unsqueeze(1))
            ranks = (scores > target_scores).sum(dim=1) + 1

            if writer is not None:
                topk_val = min(topk, scores.size(1))
                fused_scores, fused_indices = torch.topk(scores, k=topk_val, dim=1)

                for row_idx in range(scores.size(0)):
                    record = {
                        'h': int(h_ids[row_idx]),
                        'r': int(r_ids[row_idx]),
                        't': int(t_ids[row_idx].item()),
                        'rank': int(ranks[row_idx].item()),
                        'topk': fused_indices[row_idx].tolist(),
                        'topk_scores': fused_scores[row_idx].tolist(),
                    }
                    writer.write(json.dumps(record) + '\n')

            hits1 += (ranks <= 1).sum().item()
            hits3 += (ranks <= 3).sum().item()
            hits10 += (ranks <= 10).sum().item()
            mrr += (1.0 / ranks.float()).sum().item()
            mr += ranks.float().sum().item()
            total += ranks.size(0)

    if writer is not None:
        writer.close()

    return {
        'MRR': mrr / total,
        'MR': mr / total,
        'Hits@1': hits1 / total,
        'Hits@3': hits3 / total,
        'Hits@10': hits10 / total
    }
