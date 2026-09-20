import torch
from torch.utils.data import Dataset
import json
import os


class TrainTruthIndex:
    """Builds in-batch truth masks from training triples."""

    def __init__(self, train_triples):
        train_triples = torch.as_tensor(train_triples, dtype=torch.long)
        if train_triples.dim() != 2 or train_triples.size(1) not in (3, 4):
            raise ValueError(
                "Training triples must have shape (N, 3) or (N, 4) to build positives."
            )

        self.query_tails = {}
        self.temporal = train_triples.size(1) == 4
        for row in train_triples.tolist():
            if self.temporal:
                h, r, t, time_id = row
                key = (h, r, time_id)
            else:
                h, r, t = row
                key = (h, r)
            self.query_tails.setdefault(key, set()).add(t)

    def build_in_batch_truth_mask(
        self,
        head_ids,
        relation_ids,
        candidate_tail_ids,
        time_ids=None,
        device=None,
    ):
        """Returns a (batch, batch) boolean mask of which candidates are true answers."""
        head_ids = torch.as_tensor(head_ids, dtype=torch.long).reshape(-1).cpu()
        relation_ids = torch.as_tensor(
            relation_ids, dtype=torch.long
        ).reshape(-1).cpu()
        candidate_tail_ids = torch.as_tensor(
            candidate_tail_ids, dtype=torch.long
        ).reshape(-1).cpu()

        batch_size = head_ids.numel()
        if relation_ids.numel() != batch_size:
            raise ValueError(
                "head_ids and relation_ids must contain the same number of rows."
            )
        if candidate_tail_ids.numel() != batch_size:
            raise ValueError(
                "In-batch loss requires one candidate tail per query row."
            )
        if self.temporal:
            if time_ids is None:
                raise ValueError("Temporal truth masks require time_ids.")
            time_ids = torch.as_tensor(time_ids, dtype=torch.long).reshape(-1).cpu()
            if time_ids.numel() != batch_size:
                raise ValueError(
                    "time_ids must contain the same number of rows as head_ids."
                )

        candidate_columns = {}
        for column, tail_id in enumerate(candidate_tail_ids.tolist()):
            candidate_columns.setdefault(tail_id, []).append(column)

        truth_mask = torch.zeros(
            batch_size, batch_size, dtype=torch.bool
        )
        for row, (head_id, relation_id) in enumerate(zip(head_ids.tolist(), relation_ids.tolist())):
            if self.temporal:
                key = (head_id, relation_id, int(time_ids[row].item()))
            else:
                key = (head_id, relation_id)
            for tail_id in self.query_tails.get(key, ()):
                columns = candidate_columns.get(tail_id)
                if columns:
                    truth_mask[row, columns] = True

        truth_mask.fill_diagonal_(True)
        if device is not None:
            truth_mask = truth_mask.to(device)
        return truth_mask


class SnapshotEdgeBank:
    """Indexes quadruples by time_id and serves edge windows for the graph evolver."""

    def __init__(self, quadruples):
        quadruples = torch.as_tensor(quadruples, dtype=torch.long)
        if quadruples.dim() != 2 or quadruples.size(1) != 4:
            raise ValueError("SnapshotEdgeBank requires quadruples shaped (N, 4).")

        edges_by_time = {}
        for s, r, o, t in quadruples.tolist():
            edges_by_time.setdefault(t, []).append((s, r, o))
        self._sorted_times = sorted(edges_by_time)
        self._edge_tensor_by_time = {
            t: torch.tensor(edges, dtype=torch.long)
            for t, edges in edges_by_time.items()
        }

    @property
    def known_times(self):
        return self._sorted_times

    def edges_at(self, time_id):
        return self._edge_tensor_by_time[int(time_id)]

    def get_exact_snapshot_edges(self, time_id):
        """Returns the edges whose time_id exactly matches the given time."""
        return self._edge_tensor_by_time.get(int(time_id), torch.empty(0, 3, dtype=torch.long))

    def get_window_edges_by_time(self, time_id, history_len=None):
        """Returns the edges from strictly-past snapshots within the history window."""
        time_id = int(time_id)
        if history_len is not None and history_len > 0:
            lower = time_id - int(history_len)
            window_times = [t for t in self._sorted_times if lower <= t < time_id]
        else:
            window_times = [t for t in self._sorted_times if t < time_id]
        return [(t, self._edge_tensor_by_time[t]) for t in window_times]


class PackedSnapshotBatchSampler(torch.utils.data.Sampler):
    """Batches dataset rows so each batch touches a bounded number of distinct time_ids."""

    def __init__(self, dataset, max_batch_size, seed=0, drop_last_partial=False):
        self.max_batch_size = int(max_batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last_partial = bool(drop_last_partial)

        time_ids = dataset.quadruples[:, 3].tolist()
        groups = {}
        for row_idx, t in enumerate(time_ids):
            groups.setdefault(t, []).append(row_idx)
        self._group_keys = sorted(groups)
        self._groups = groups
        self._cached_len_epoch = None
        self._cached_len_count = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _make_batches(self):
        """Builds this epoch's batches by shuffling and packing time-groups."""
        order_generator = torch.Generator()
        order_generator.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self._group_keys), generator=order_generator).tolist()
        ordered_keys = [self._group_keys[i] for i in perm]

        batches = []
        current = []
        for key in ordered_keys:
            group_rows = self._groups[key]
            if current and len(current) + len(group_rows) > self.max_batch_size:
                batches.append(current)
                current = []
            current.extend(group_rows)
            if len(current) >= self.max_batch_size:
                batches.append(current)
                current = []
        if current and not self.drop_last_partial:
            batches.append(current)

        row_generator = torch.Generator()
        row_generator.manual_seed(self.seed + self.epoch + 1)
        shuffled_batches = []
        for batch in batches:
            idx = torch.randperm(len(batch), generator=row_generator).tolist()
            shuffled_batches.append([batch[i] for i in idx])

        batch_order_generator = torch.Generator()
        batch_order_generator.manual_seed(self.seed + self.epoch + 2)
        batch_perm = torch.randperm(len(shuffled_batches), generator=batch_order_generator).tolist()
        return [shuffled_batches[i] for i in batch_perm]

    def __iter__(self):
        return iter(self._make_batches())

    def __len__(self):
        if self._cached_len_epoch != self.epoch:
            self._cached_len_count = len(self._make_batches())
            self._cached_len_epoch = self.epoch
        return self._cached_len_count


class TemporalGWMDataset(Dataset):
    """Dataset of temporal quadruples with per-query historical context."""

    def __init__(self, data_dir, split='train', context_k=10, history_len=3, history_quadruples=None):
        self.data_dir = data_dir
        self.split = split
        self.context_k = int(context_k)
        self.history_len = int(history_len)

        with open(os.path.join(data_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.num_entities = len(json.load(f))
        with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.num_relations = len(json.load(f))

        quad_path = os.path.join(data_dir, f'{split}_quadruples.pt')
        if not os.path.exists(quad_path):
            raise FileNotFoundError(f"Quadruple tensor not found: {quad_path}")
        self.quadruples = torch.load(quad_path, map_location='cpu').long()
        if self.quadruples.dim() != 2 or self.quadruples.size(1) != 4:
            raise ValueError(
                f"Expected quadruples with shape (N, 4), got {tuple(self.quadruples.shape)}"
            )

        if history_quadruples is not None:
            train_quads = torch.as_tensor(history_quadruples, dtype=torch.long)
        else:
            train_path = os.path.join(data_dir, 'train_quadruples.pt')
            train_quads = torch.load(train_path, map_location='cpu').long()
        self.history_by_subject = {}
        for s, r, o, time_id in train_quads.tolist():
            self.history_by_subject.setdefault(s, []).append((time_id, r, o))
        for subject in self.history_by_subject:
            self.history_by_subject[subject].sort(key=lambda item: (item[0], item[1], item[2]))

        self.relations_by_subject_time = {}
        for split_name in ('train', 'valid', 'test'):
            split_path = os.path.join(data_dir, f'{split_name}_quadruples.pt')
            if not os.path.exists(split_path):
                continue
            split_quads = torch.load(split_path, map_location='cpu').long()
            for s, r, _, time_id in split_quads.tolist():
                self.relations_by_subject_time.setdefault((s, time_id), set()).add(r)

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        """Returns one query quadruple with its historical context facts."""
        h, r, t, time_id = self.quadruples[idx]
        h_idx = int(h.item())
        r_idx = int(r.item())
        t_idx = int(t.item())
        q_time = int(time_id.item())

        min_time = q_time - self.history_len if self.history_len > 0 else None
        candidates = []
        for hist_time, hist_rel, hist_ent in self.history_by_subject.get(h_idx, []):
            if hist_time >= q_time:
                break
            if min_time is not None and hist_time < min_time:
                continue
            if hist_rel == r_idx and hist_ent == t_idx and hist_time == q_time:
                continue
            candidates.append((hist_time, hist_rel, hist_ent))

        if self.context_k > 0:
            candidates = candidates[-self.context_k:]

        if candidates:
            ctx_time_ids = torch.tensor([x[0] for x in candidates], dtype=torch.long)
            ctx_relation_ids = torch.tensor([x[1] for x in candidates], dtype=torch.long)
            ctx_entity_ids = torch.tensor([x[2] for x in candidates], dtype=torch.long)
            ctx_mask = torch.ones(len(candidates), dtype=torch.bool)
        else:
            ctx_time_ids = torch.zeros(0, dtype=torch.long)
            ctx_relation_ids = torch.zeros(0, dtype=torch.long)
            ctx_entity_ids = torch.zeros(0, dtype=torch.long)
            ctx_mask = torch.zeros(0, dtype=torch.bool)

        ref_relations = sorted(self.relations_by_subject_time.get((h_idx, q_time), {r_idx}))
        reference_relation_ids = torch.tensor(ref_relations, dtype=torch.long)

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'time_id': time_id.long(),
            'context_entity_ids': ctx_entity_ids,
            'context_relation_ids': ctx_relation_ids,
            'context_time_ids': ctx_time_ids,
            'context_mask': ctx_mask,
            'reference_relation_ids': reference_relation_ids,
        }


class GWMDataset(Dataset):
    """Dataset of static triples with precomputed context neighbors."""

    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(data_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.num_entities = len(json.load(f))
        with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.num_relations = len(json.load(f))

        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        if not os.path.exists(triples_path):
            if split == 'valid':
                triples_path = os.path.join(data_dir, 'dev_triples.pt')
        if not os.path.exists(triples_path):
            raise FileNotFoundError(f"Triple tensor not found: {triples_path}")

        self.triples = torch.load(triples_path, map_location='cpu').long()
        if self.triples.dim() != 2 or self.triples.size(1) != 3:
            raise ValueError(
                f"Expected triples with shape (N, 3), got {tuple(self.triples.shape)}"
            )

        context_pack_path = os.path.join(data_dir, 'context_neighbors.pt')

        self.context_entity_ids = None
        self.context_relation_ids = None
        self.context_mask = None
        self.context_pad_value = -1

        if os.path.exists(context_pack_path):
            context_pack = torch.load(context_pack_path, map_location='cpu')
            self.context_entity_ids = context_pack['entity_ids'].long()
            self.context_relation_ids = context_pack['relation_ids'].long()
            self.context_mask = context_pack['mask'].bool()
            self.context_pad_value = int(context_pack.get('pad_value', -1))
            expected_shape = self.context_entity_ids.shape
            if (
                self.context_entity_ids.dim() != 2
                or self.context_relation_ids.shape != expected_shape
                or self.context_mask.shape != expected_shape
            ):
                raise ValueError(
                    "Context entity IDs, relation IDs, and mask must share "
                    "the same rank-2 shape."
                )
            if self.context_entity_ids.size(0) != self.num_entities:
                raise ValueError(
                    "Context artifact row count must equal the entity vocabulary size."
                )
            valid_entities = self.context_entity_ids[self.context_mask]
            valid_relations = self.context_relation_ids[self.context_mask]
            if valid_entities.numel() and (
                valid_entities.min() < 0
                or valid_entities.max() >= self.num_entities
            ):
                raise ValueError("Context artifact contains invalid entity IDs.")
            if valid_relations.numel() and (
                valid_relations.min() < 0
                or valid_relations.max() >= self.num_relations
            ):
                raise ValueError("Context artifact contains invalid relation IDs.")
        else:
            raise FileNotFoundError(
                "Error: context files not found "
                "(expected context_neighbors.pt)."
            )

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """Returns one query triple with its context, excluding the answer edge."""
        h, r, t = self.triples[idx]
        h_idx = int(h.item())

        if self.context_entity_ids is not None:
            ctx_entity_ids = self.context_entity_ids[h_idx]
            ctx_relation_ids = self.context_relation_ids[h_idx]
            ctx_mask = self.context_mask[h_idx].clone()

            target_edge = (
                ctx_entity_ids.eq(int(t.item()))
                & ctx_relation_ids.eq(int(r.item()))
            )
            ctx_mask &= ~target_edge
        else:
            ctx_entity_ids = torch.zeros(0, dtype=torch.long)
            ctx_relation_ids = torch.zeros(0, dtype=torch.long)
            ctx_mask = torch.zeros(0, dtype=torch.bool)

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'context_entity_ids': ctx_entity_ids.long(),
            'context_relation_ids': ctx_relation_ids.long(),
            'context_mask': ctx_mask.bool(),
        }

class CollateFN:
    """Collates a batch into padded/flattened tensors for the model."""
    def __call__(self, batch):
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])
        has_time = 'time_id' in batch[0]
        time_ids = torch.stack([b['time_id'] for b in batch]) if has_time else None

        context_entity_chunks = []
        context_relation_chunks = []
        context_time_chunks = []
        context_batch_chunks = []
        reference_relation_chunks = []
        reference_batch_chunks = []
        for sample_idx, item in enumerate(batch):
            ent_ids = item['context_entity_ids']
            rel_ids = item['context_relation_ids']
            ctx_time_ids = item.get('context_time_ids')
            mask = item['context_mask'].bool()

            if ent_ids.dim() != 1 or rel_ids.dim() != 1 or mask.dim() != 1:
                raise ValueError("Each context row must be one-dimensional.")
            if not (ent_ids.numel() == rel_ids.numel() == mask.numel()):
                raise ValueError("Context entity, relation, and mask lengths differ.")
            if ctx_time_ids is not None and ctx_time_ids.numel() != ent_ids.numel():
                raise ValueError("Context time and entity lengths differ.")

            valid_ent = ent_ids[mask]
            valid_rel = rel_ids[mask]
            valid_time = ctx_time_ids[mask] if ctx_time_ids is not None else None

            valid_pair_mask = (valid_ent >= 0) & (valid_rel >= 0)
            valid_ent = valid_ent[valid_pair_mask]
            valid_rel = valid_rel[valid_pair_mask]
            if valid_time is not None:
                valid_time = valid_time[valid_pair_mask]

            if valid_ent.numel() > 0:
                context_entity_chunks.append(valid_ent.long())
                context_relation_chunks.append(valid_rel.long())
                if valid_time is not None:
                    context_time_chunks.append(valid_time.long())
                context_batch_chunks.append(torch.full((valid_ent.numel(),), sample_idx, dtype=torch.long))

            ref_rel_ids = item.get('reference_relation_ids')
            if ref_rel_ids is not None and ref_rel_ids.numel() > 0:
                reference_relation_chunks.append(ref_rel_ids.long())
                reference_batch_chunks.append(
                    torch.full((ref_rel_ids.numel(),), sample_idx, dtype=torch.long)
                )

        if context_entity_chunks:
            context_entity_ids = torch.cat(context_entity_chunks, dim=0)
            context_relation_ids = torch.cat(context_relation_chunks, dim=0)
            context_batch_index = torch.cat(context_batch_chunks, dim=0)
        else:
            context_entity_ids = torch.zeros(0, dtype=torch.long)
            context_relation_ids = torch.zeros(0, dtype=torch.long)
            context_batch_index = torch.zeros(0, dtype=torch.long)

        h_batch = {'id': h_ids}
        t_batch = {'id': t_ids}
        if has_time:
            h_batch['time_id'] = time_ids
            t_batch['time_id'] = time_ids

        context_batch = {
            'id': context_entity_ids,
            'rel_id': context_relation_ids,
            'batch_index': context_batch_index,
        }
        if context_time_chunks:
            context_batch['time_id'] = torch.cat(context_time_chunks, dim=0)
        elif has_time:
            context_batch['time_id'] = torch.zeros(0, dtype=torch.long)
        if reference_relation_chunks:
            context_batch['reference_rel_id'] = torch.cat(reference_relation_chunks, dim=0)
            context_batch['reference_batch_index'] = torch.cat(reference_batch_chunks, dim=0)

        return {
            'h_batch': h_batch,
            'r_batch': {'id': r_ids},
            't_batch': t_batch,
            'time_batch': {'id': time_ids} if has_time else None,
            'context_batch': context_batch,
        }
