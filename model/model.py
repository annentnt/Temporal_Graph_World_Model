import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.graph_evolver import GraphWorldStateEvolver


class ContextAggregator(nn.Module):
    """Pool relation-composed facts, add the head residual, and normalize."""

    SUPPORTED_REDUCTIONS = {'mean', 'max'}

    def __init__(self, hidden_dim, reduction='mean'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reduction = str(reduction).lower()
        if self.reduction not in self.SUPPORTED_REDUCTIONS:
            supported = ', '.join(sorted(self.SUPPORTED_REDUCTIONS))
            raise ValueError(
                f"Unsupported context aggregation '{reduction}'. "
                f"Expected one of: {supported}."
            )

        self.norm = nn.LayerNorm(hidden_dim)

    def _aggregate(self, messages, batch_index, batch_size, reference):
        aggregated = torch.zeros_like(reference)
        if messages.numel() == 0:
            return aggregated

        if self.reduction == 'mean':
            aggregated.index_add_(0, batch_index, messages)
            counts = torch.zeros(
                batch_size,
                1,
                device=reference.device,
                dtype=reference.dtype,
            )
            counts.index_add_(
                0,
                batch_index,
                torch.ones(
                    messages.size(0),
                    1,
                    device=reference.device,
                    dtype=reference.dtype,
                ),
            )
            return aggregated / counts.clamp_min(1.0)

        aggregated.fill_(float('-inf'))
        expanded_index = batch_index.unsqueeze(-1).expand_as(messages)
        aggregated.scatter_reduce_(
            0,
            expanded_index,
            messages,
            reduce='amax',
            include_self=True,
        )
        return torch.where(
            torch.isfinite(aggregated),
            aggregated,
            torch.zeros_like(aggregated),
        )

    def forward(self, head_feat, nbr_entity_feat, nbr_relation_feat, nbr_batch_index):
        """Aggregates context facts for each head in the batch."""
        if nbr_entity_feat.shape != nbr_relation_feat.shape:
            raise ValueError(
                "Context entity and relation features must have identical shapes."
            )
        if nbr_batch_index.dim() != 1:
            raise ValueError("Context batch indices must be one-dimensional.")
        if nbr_batch_index.numel() != nbr_entity_feat.size(0):
            raise ValueError(
                "Each context fact must have one corresponding batch index."
            )
        if nbr_batch_index.numel() and (
            nbr_batch_index.min() < 0
            or nbr_batch_index.max() >= head_feat.size(0)
        ):
            raise ValueError("Context batch index is outside the current batch.")

        composed_facts = nbr_entity_feat * nbr_relation_feat
        agg = self._aggregate(
            composed_facts,
            nbr_batch_index,
            head_feat.size(0),
            head_feat,
        )
        return self.norm(head_feat + agg)


class MLPAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class GatedFusion(nn.Module):
    """Project two modalities into one space and combine them feature-wise."""

    def __init__(self, text_dim, struct_dim, fusion_dim, dropout=0.0):
        super().__init__()
        self.text_layer_norm = nn.LayerNorm(text_dim)
        self.struct_layer_norm = nn.LayerNorm(struct_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.struct_projection = nn.Linear(struct_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def struct_only(self, struct_features):
        """Fuses using only the structural features, bypassing text and the gate."""
        struct_projected = self.struct_projection(
            self.struct_layer_norm(struct_features)
        )
        return self.output_norm(self.dropout(struct_projected))

    def forward(self, text_features, struct_features):
        text_projected = self.text_projection(
            self.text_layer_norm(text_features)
        )
        struct_projected = self.struct_projection(
            self.struct_layer_norm(struct_features)
        )
        gate = self.gate(torch.cat([text_projected, struct_projected], dim=-1))
        fused = gate * text_projected + (1.0 - gate) * struct_projected
        return self.output_norm(self.dropout(fused)), gate


class ConvTransEDecoder(nn.Module):
    """ConvTransE decoder over a query state, relation state, and all candidates."""

    def __init__(self, embedding_dim, dropout=0.0, channels=50, kernel_size=3):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        self.conv = nn.Conv1d(
            2,
            channels,
            kernel_size,
            stride=1,
            padding=int(math.floor(kernel_size / 2)),
        )
        self.fc = nn.Linear(self.embedding_dim * channels, self.embedding_dim)

    def forward(self, query_vectors, relation_vectors, candidate_vectors):
        batch_size = query_vectors.size(0)
        stacked_inputs = torch.stack([query_vectors, relation_vectors], dim=1)
        x = self.bn0(stacked_inputs)
        x = self.dropout1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        x = self.dropout3(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        return torch.mm(x, torch.tanh(candidate_vectors).t())


class ExplicitTemporalEntityEncoder(nn.Module):
    """Re-Temp-style static + trend + seasonal entity representation."""

    def __init__(
        self,
        num_entities,
        hidden_dim,
        alpha=0.5,
        time_scale=1.0,
        dropout=0.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.time_scale = float(time_scale)
        self.trend = nn.Embedding(num_entities, hidden_dim)
        self.seasonal = nn.Embedding(num_entities, hidden_dim)
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        nn.init.normal_(self.trend.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.seasonal.weight, mean=0.0, std=0.02)

    def forward(self, static_features, entity_ids, time_ids):
        if time_ids is None:
            return static_features
        if time_ids.dim() != 1:
            time_ids = time_ids.reshape(-1)
        if time_ids.numel() != entity_ids.numel():
            raise ValueError("time_ids and entity_ids must have the same length.")

        t = time_ids.to(
            device=static_features.device,
            dtype=static_features.dtype,
        ).unsqueeze(-1)
        t = t * self.time_scale
        trend = self.trend(entity_ids)
        seasonal = self.seasonal(entity_ids)
        dynamic = (
            self.alpha * trend * t
            + (1.0 - self.alpha) * torch.sin(2.0 * math.pi * seasonal * t)
        )
        return self.mix(torch.cat([static_features, dynamic], dim=-1))


class SnapshotCompGCNEncoder(nn.Module):
    """Structural encoder that pools a query's historical context edges per snapshot."""

    def __init__(self, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = int(num_layers)
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(self.num_layers)
        )
        self.loop_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(self.num_layers)
        )
        self.norm_layers = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def _pool_snapshot_messages(self, messages, batch_index, time_id, batch_size):
        time_mod = int(time_id.max().item()) + 1 if time_id.numel() else 1
        group_key = batch_index * time_mod + time_id
        unique_key, inverse = torch.unique(
            group_key,
            sorted=True,
            return_inverse=True,
        )

        pooled = torch.zeros(
            unique_key.numel(),
            messages.size(1),
            device=messages.device,
            dtype=messages.dtype,
        )
        pooled.index_add_(0, inverse, messages)
        counts = torch.zeros(
            unique_key.numel(),
            1,
            device=messages.device,
            dtype=messages.dtype,
        )
        counts.index_add_(
            0,
            inverse,
            torch.ones(messages.size(0), 1, device=messages.device, dtype=messages.dtype),
        )
        pooled = pooled / counts.clamp_min(1.0)
        snapshot_batch_index = unique_key // time_mod
        snapshot_time_id = unique_key % time_mod
        if snapshot_batch_index.numel() and (
            snapshot_batch_index.min() < 0
            or snapshot_batch_index.max() >= batch_size
        ):
            raise ValueError("Snapshot batch index is outside the current batch.")
        return pooled, snapshot_batch_index, snapshot_time_id

    def forward(
        self,
        head_feat,
        nbr_entity_feat,
        nbr_relation_feat,
        nbr_batch_index,
        nbr_time_id,
    ):
        if nbr_entity_feat.shape != nbr_relation_feat.shape:
            raise ValueError(
                "Snapshot context entity and relation features must have identical shapes."
            )
        if nbr_batch_index.numel() != nbr_entity_feat.size(0):
            raise ValueError("Each snapshot edge needs one batch index.")
        if nbr_time_id.numel() != nbr_entity_feat.size(0):
            raise ValueError("Each snapshot edge needs one time id.")
        if nbr_entity_feat.numel() == 0:
            return (
                torch.zeros(0, self.hidden_dim, device=head_feat.device, dtype=head_feat.dtype),
                torch.zeros(0, device=head_feat.device, dtype=torch.long),
                torch.zeros(0, device=head_feat.device, dtype=torch.long),
            )

        snapshot_state = nbr_entity_feat + nbr_relation_feat
        snapshot_batch_index = None
        snapshot_time_id = None
        for layer_idx, (message_layer, loop_layer, norm_layer) in enumerate(
            zip(self.message_layers, self.loop_layers, self.norm_layers)
        ):
            messages = message_layer(snapshot_state)
            if layer_idx == 0:
                batch_index = nbr_batch_index
                time_id = nbr_time_id
            else:
                batch_index = snapshot_batch_index
                time_id = snapshot_time_id

            pooled, snapshot_batch_index, snapshot_time_id = self._pool_snapshot_messages(
                messages,
                batch_index,
                time_id,
                batch_size=head_feat.size(0),
            )
            loop = loop_layer(head_feat[snapshot_batch_index])
            snapshot_state = norm_layer(F.gelu(pooled + loop))
            snapshot_state = self.dropout(snapshot_state)

        return snapshot_state, snapshot_batch_index, snapshot_time_id


class RelationAwareTemporalContextAggregator(nn.Module):
    """Attend over snapshot structural states using a relation reference vector."""

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        head_feat,
        relation_ref_feat,
        snapshot_state,
        snapshot_batch_index,
    ):
        if snapshot_batch_index.numel() != snapshot_state.size(0):
            raise ValueError("Each snapshot state needs one batch index.")
        if snapshot_state.numel() == 0:
            return self.norm(head_feat)

        batch_size = head_feat.size(0)
        query = self.query_proj(relation_ref_feat[snapshot_batch_index])
        key = self.key_proj(snapshot_state)
        scores = self.score_proj(torch.tanh(query + key)).squeeze(-1)

        attended = torch.zeros_like(head_feat)
        for sample_idx in range(batch_size):
            mask = snapshot_batch_index == sample_idx
            if not torch.any(mask):
                continue
            weights = torch.softmax(scores[mask], dim=0).unsqueeze(-1)
            attended[sample_idx] = torch.sum(weights * snapshot_state[mask], dim=0)

        return self.norm(head_feat + self.dropout(attended))


class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = float(getattr(config, 'dropout'))
        self.adapter_dropout = float(getattr(config, 'adapter_dropout', self.dropout))
        # Ablation switch: use only the structural representation, skipping
        # the text branch and gate.
        self.disable_text_modality = bool(getattr(config, 'disable_text_modality', False))

        # 1. Text Components (Entity/Relation Embeddings)
        self.text_emb_dim = int(getattr(config, 'text_emb_dim'))
        self.text_ent_embs = nn.Embedding(config.num_entities, self.text_emb_dim)
        self.text_rel_embs = nn.Embedding(config.num_relations, self.text_emb_dim)

        self.text_adapter = MLPAdapter(self.text_emb_dim,
            int(getattr(config, 'text_adapter_dim')),
            dropout=self.adapter_dropout
            )

        # 2. Structural Components (Entity/Relation Embeddings)
        self.struct_emb_dim = int(getattr(config, 'struct_emb_dim'))
        self.struct_ent_embs = nn.Embedding(config.num_entities, self.struct_emb_dim)
        self.struct_rel_embs = nn.Embedding(config.num_relations, self.struct_emb_dim)

        self.struct_adapter = MLPAdapter(
            self.struct_emb_dim,
            int(getattr(config, 'struct_adapter_dim')),
            dropout=self.adapter_dropout
            )

        # 3. Early Fusion and Shared Dynamics
        self.fusion_dim = int(getattr(config, 'fusion_dim'))
        self.entity_fusion = GatedFusion(
            self.text_emb_dim,
            self.struct_emb_dim,
            self.fusion_dim,
            dropout=self.dropout,
        )
        self.relation_fusion = GatedFusion(
            self.text_emb_dim,
            self.struct_emb_dim,
            self.fusion_dim,
            dropout=self.dropout,
        )

        self.context_agg = str(getattr(config, 'context_agg', 'mean')).lower()
        self.fused_context_aggregator = ContextAggregator(
            hidden_dim=self.fusion_dim,
            reduction=self.context_agg,
        )
        self.temporal_enabled = bool(getattr(config, 'temporal_enabled', False))
        if self.temporal_enabled:
            self.temporal_entity_encoder = ExplicitTemporalEntityEncoder(
                num_entities=config.num_entities,
                hidden_dim=self.fusion_dim,
                alpha=float(getattr(config, 'temporal_alpha', 0.5)),
                time_scale=float(getattr(config, 'temporal_time_scale', 1.0)),
                dropout=self.dropout,
            )
            self.snapshot_structural_encoder = SnapshotCompGCNEncoder(
                hidden_dim=self.fusion_dim,
                num_layers=int(getattr(config, 'snapshot_gnn_layers', 1)),
                dropout=self.dropout,
            )
            self.temporal_context_aggregator = RelationAwareTemporalContextAggregator(
                hidden_dim=self.fusion_dim,
                dropout=self.dropout,
            )
            self.graph_evolver = GraphWorldStateEvolver(
                dim=self.struct_emb_dim,
                num_layers=int(
                    getattr(
                        config,
                        'graph_evolver_layers',
                        getattr(config, 'snapshot_gnn_layers', 2),
                    )
                ),
                dropout=self.dropout,
            )
        else:
            self.temporal_entity_encoder = None
            self.snapshot_structural_encoder = None
            self.temporal_context_aggregator = None
            self.graph_evolver = None
        self.fused_h0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.fused_c0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)

        dynamics_layers = int(getattr(config, 'dynamics_layers', 1))
        self.fused_lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim,
            num_layers=dynamics_layers,
            batch_first=True,
            dropout=self.dropout if dynamics_layers > 1 else 0.0,
        )
        self.fused_output_projection = nn.Linear(
            self.fusion_dim,
            self.fusion_dim,
        )

        self.temperature = float(getattr(config, 'temperature'))
        self.decoder_name = str(getattr(config, 'decoder', 'dot')).lower()
        self.cache_static_candidates = bool(getattr(config, 'cache_static_candidates', False))
        if self.cache_static_candidates and self.temporal_enabled:
            raise ValueError(
                "cache_static_candidates is incompatible with temporal_enabled=True: "
                "candidates are time-conditioned via the graph world-state evolver "
                "and cannot be cached as one static, time-independent table."
            )
        self._static_candidate_cache = None
        if self.decoder_name == 'convtranse':
            self.decoder = ConvTransEDecoder(
                embedding_dim=self.fusion_dim,
                dropout=self.dropout,
                channels=int(getattr(config, 'convtranse_channels', 50)),
                kernel_size=int(getattr(config, 'convtranse_kernel_size', 3)),
            )
        elif self.decoder_name in {'dot', 'contrastive'}:
            self.decoder = None
        else:
            raise ValueError(f"Unsupported decoder: {self.decoder_name}")

    def _prepare_context_batch(self, context_batch):
        context_entity_ids = context_batch['id']
        context_relation_ids = context_batch.get('rel_id')
        context_batch_index = context_batch.get('batch_index')
        if context_relation_ids is None or context_batch_index is None:
            raise ValueError(
                "context_batch requires 'id', 'rel_id', and 'batch_index'."
            )
        if (
            context_entity_ids.dim() != 1
            or context_relation_ids.dim() != 1
            or context_batch_index.dim() != 1
        ):
            raise ValueError("Ragged context tensors must all be one-dimensional.")
        if not (
            context_entity_ids.numel()
            == context_relation_ids.numel()
            == context_batch_index.numel()
        ):
            raise ValueError("Ragged context tensors must have equal lengths.")
        return context_entity_ids, context_relation_ids, context_batch_index

    def _gather_graph_struct(self, entity_ids, time_ids, graph_state_by_time):
        """Gathers per-row structural features from the time-conditioned entity tables."""
        if time_ids is None:
            raise ValueError("graph_state_by_time requires matching time_ids.")
        time_ids = time_ids.reshape(-1)
        if time_ids.numel() != entity_ids.numel():
            raise ValueError("time_ids and entity_ids must align for graph-state gather.")

        out = torch.empty(
            entity_ids.numel(),
            self.struct_emb_dim,
            device=entity_ids.device,
            dtype=self.struct_ent_embs.weight.dtype,
        )
        for t in torch.unique(time_ids).tolist():
            mask = time_ids == t
            state_table = graph_state_by_time[int(t)]
            out[mask] = state_table[entity_ids[mask]]
        return out

    def _encode_entity(self, entity_ids, time_ids=None, graph_state_by_time=None):
        if graph_state_by_time:
            ent_struct_raw = self._gather_graph_struct(entity_ids, time_ids, graph_state_by_time)
        else:
            ent_struct_raw = self.struct_ent_embs(entity_ids)
        ent_struct = self.struct_adapter(ent_struct_raw)
        if self.disable_text_modality:
            ent_fused = self.entity_fusion.struct_only(ent_struct)
        else:
            ent_text = self.text_adapter(self.text_ent_embs(entity_ids))
            ent_fused, _ = self.entity_fusion(ent_text, ent_struct)
        if self.temporal_entity_encoder is not None:
            ent_fused = self.temporal_entity_encoder(
                ent_fused,
                entity_ids,
                time_ids,
            )
        return ent_fused

    def build_graph_world_states(self, time_ids, edge_bank, history_len=None):
        """Computes the full-graph evolved structural entity table for each requested time_id."""
        if self.graph_evolver is None:
            return None
        if history_len is None:
            history_len = int(getattr(self.config, 'history_len', 3))

        unique_times = sorted({int(t) for t in torch.as_tensor(time_ids).reshape(-1).tolist()})
        if not unique_times:
            return {}

        base_state = self.struct_ent_embs.weight
        rel_table = self.struct_rel_embs.weight
        num_entities = int(self.config.num_entities)

        # Each time_id's window is evolved independently, re-seeded from base_state.
        num_base_rels = int(self.config.num_relations) // 2

        result = {}
        for t in unique_times:
            window = edge_bank.get_window_edges_by_time(t, history_len)
            if not window:
                result[t] = base_state
                continue
            raw_snapshot_edges = edge_bank.get_exact_snapshot_edges(t)
            if raw_snapshot_edges.numel() > 0:
                own_snapshot_edges = raw_snapshot_edges[
                    raw_snapshot_edges[:, 1] < num_base_rels
                ]
            else:
                own_snapshot_edges = raw_snapshot_edges
            snapshot_states = self.graph_evolver(
                base_entity_state=base_state,
                rel_emb_table=rel_table,
                time_ordered_edges=window,
                num_entities=num_entities,
                same_snapshot_edges=own_snapshot_edges,
            )
            result[t] = snapshot_states[window[-1][0]]
        return result

    def _encode_relation(self, relation_ids):
        rel_struct = self.struct_adapter(self.struct_rel_embs(relation_ids))
        if self.disable_text_modality:
            return self.relation_fusion.struct_only(rel_struct)
        rel_text = self.text_adapter(self.text_rel_embs(relation_ids))
        rel_fused, _ = self.relation_fusion(rel_text, rel_struct)
        return rel_fused

    def _encode_relation_reference(self, context_batch, fallback_relation_feat, batch_size):
        reference_relation_ids = context_batch.get('reference_rel_id')
        reference_batch_index = context_batch.get('reference_batch_index')
        if reference_relation_ids is None or reference_batch_index is None:
            return fallback_relation_feat
        if reference_relation_ids.numel() == 0:
            return fallback_relation_feat

        reference_rel_feat = self._encode_relation(reference_relation_ids)
        reference = torch.zeros_like(fallback_relation_feat)
        reference.index_add_(0, reference_batch_index, reference_rel_feat)
        counts = torch.zeros(
            batch_size,
            1,
            device=fallback_relation_feat.device,
            dtype=fallback_relation_feat.dtype,
        )
        counts.index_add_(
            0,
            reference_batch_index,
            torch.ones(
                reference_rel_feat.size(0),
                1,
                device=fallback_relation_feat.device,
                dtype=fallback_relation_feat.dtype,
            ),
        )
        has_reference = counts.squeeze(-1) > 0
        reference = reference / counts.clamp_min(1.0)
        return torch.where(has_reference.unsqueeze(-1), reference, fallback_relation_feat)

    def _run_dynamics(self, world_state, head_emb, relation_emb, lstm, h0_proj, c0_proj):
        """Runs the query LSTM over head and relation embeddings, seeded from world_state."""
        h_0 = torch.tanh(h0_proj(world_state))
        c_0 = torch.tanh(c0_proj(world_state))

        num_layers = lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        _, (h_n, _) = lstm(torch.stack([head_emb, relation_emb], dim=1), (h_0_lstm, c_0_lstm))
        query_vector = h_n[-1]
        return query_vector

    def _load_embedding_tensor(self, source, expected_rows, name):
        if isinstance(source, str):
            loaded = torch.load(source, map_location='cpu')
        elif torch.is_tensor(source):
            loaded = source.detach().cpu()
        else:
            raise TypeError(f"Unsupported {name} cache source: {type(source)}")

        if isinstance(loaded, dict):
            if 'embeddings' in loaded:
                loaded = loaded['embeddings']
            elif 'tensor' in loaded:
                loaded = loaded['tensor']
            else:
                raise ValueError(f"{name} cache dict must contain 'embeddings' or 'tensor'.")

        if not torch.is_tensor(loaded):
            raise TypeError(f"{name} cache must resolve to a torch.Tensor.")

        loaded = loaded.float().contiguous()
        if loaded.dim() != 2:
            raise ValueError(f"{name} cache must be rank-2. Got shape {tuple(loaded.shape)}")
        if loaded.size(0) != expected_rows:
            raise ValueError(
                f"{name} cache row count mismatch. Expected {expected_rows}, got {loaded.size(0)}"
            )
        return loaded

    def load_embeddings(self, entity_source, relation_source, kind='text', freeze=False):
        if kind == 'text':
            entity_table = self.text_ent_embs
            relation_table = self.text_rel_embs
            expected_dim = self.text_emb_dim
            entity_name = 'text_entity'
            relation_name = 'text_relation'
        elif kind == 'structural':
            entity_table = self.struct_ent_embs
            relation_table = self.struct_rel_embs
            expected_dim = self.struct_emb_dim
            entity_name = 'structural_entity'
            relation_name = 'structural_relation'
        else:
            raise ValueError(f"Unsupported embedding kind: {kind}")

        entity_cache = self._load_embedding_tensor(
            source=entity_source,
            expected_rows=entity_table.num_embeddings,
            name=entity_name,
        )
        relation_cache = self._load_embedding_tensor(
            source=relation_source,
            expected_rows=relation_table.num_embeddings,
            name=relation_name,
        )

        if entity_cache.size(1) != relation_cache.size(1):
            raise ValueError(
                f"{entity_name} and {relation_name} embeddings must share the same embedding dimension. "
                f"Got {entity_cache.size(1)} and {relation_cache.size(1)}"
            )

        if entity_cache.size(1) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {expected_dim}, got {entity_cache.size(1)}"
            )

        entity_table.weight.data.copy_(entity_cache)
        relation_table.weight.data.copy_(relation_cache)

        if freeze:
            entity_table.weight.requires_grad = False
            relation_table.weight.requires_grad = False

    @staticmethod
    def _filtered_in_batch_contrastive_loss(
        scores,
        truth_mask=None,
    ):
        batch_size = scores.size(0)
        if scores.dim() != 2 or scores.size(1) != batch_size:
            raise ValueError(
                "In-batch contrastive scores must have shape (B, B)."
            )

        if truth_mask is None:
            truth_mask = torch.eye(
                batch_size, dtype=torch.bool, device=scores.device
            )
        else:
            if truth_mask.shape != scores.shape:
                raise ValueError(
                    "truth_mask must have the same shape as scores."
                )
            truth_mask = truth_mask.to(
                device=scores.device, dtype=torch.bool
            )

        diagonal = torch.eye(
            batch_size, dtype=torch.bool, device=scores.device
        )
        denominator_mask = (~truth_mask) | diagonal
        filtered_scores = scores.masked_fill(
            ~denominator_mask, float('-inf')
        )
        labels = torch.arange(batch_size, device=scores.device)
        return F.cross_entropy(
            filtered_scores,
            labels,
            reduction='none',
        )

    def encode_query(self, h_batch, r_batch, context_batch, graph_state_by_time=None):
        h_time = h_batch.get('time_id')
        h_fused = self._encode_entity(h_batch['id'], h_time, graph_state_by_time)
        r_fused = self._encode_relation(r_batch['id'])

        flat_context_entity_ids, flat_context_relation_ids, context_batch_index = self._prepare_context_batch(context_batch)
        ctx_time = context_batch.get('time_id')
        ctx_ent_fused = self._encode_entity(flat_context_entity_ids, ctx_time, graph_state_by_time)
        ctx_rel_fused = self._encode_relation(flat_context_relation_ids)

        if self.temporal_context_aggregator is not None and ctx_time is not None:
            relation_ref = self._encode_relation_reference(
                context_batch,
                fallback_relation_feat=r_fused,
                batch_size=h_fused.size(0),
            )
            snapshot_state, snapshot_batch_index, _ = self.snapshot_structural_encoder(
                head_feat=h_fused,
                nbr_entity_feat=ctx_ent_fused,
                nbr_relation_feat=ctx_rel_fused,
                nbr_batch_index=context_batch_index,
                nbr_time_id=ctx_time,
            )
            world_state = self.temporal_context_aggregator(
                head_feat=h_fused,
                relation_ref_feat=relation_ref,
                snapshot_state=snapshot_state,
                snapshot_batch_index=snapshot_batch_index,
            )
        else:
            world_state = self.fused_context_aggregator(
                head_feat=h_fused,
                nbr_entity_feat=ctx_ent_fused,
                nbr_relation_feat=ctx_rel_fused,
                nbr_batch_index=context_batch_index,
            )
        query = self._run_dynamics(
            world_state,
            h_fused,
            r_fused,
            self.fused_lstm,
            self.fused_h0_projection,
            self.fused_c0_projection,
        )
        query = F.normalize(self.fused_output_projection(query), p=2, dim=1)
        return query, r_fused

    def forward(self, h_batch, r_batch, context_batch, graph_state_by_time=None):
        query, _ = self.encode_query(h_batch, r_batch, context_batch, graph_state_by_time=graph_state_by_time)
        return query

    def encode_target(self, t_batch, graph_state_by_time=None):
        time_ids = t_batch.get('time_id')
        t_fused = self._encode_entity(t_batch['id'], time_ids, graph_state_by_time)
        return F.normalize(
            self.fused_output_projection(t_fused),
            p=2,
            dim=1,
        )

    def encode_static_candidates(self, entity_ids):
        """Encodes candidates for the non-temporal path (falls back to static embeddings)."""
        return self.encode_target({'id': entity_ids})

    def clear_static_candidate_cache(self):
        self._static_candidate_cache = None

    def refresh_static_candidate_cache(self, device=None):
        assert not self.temporal_enabled, (
            "refresh_static_candidate_cache caches one time-independent table "
            "and must not be used when temporal_enabled=True."
        )
        if device is None:
            device = next(self.parameters()).device
        entity_ids = torch.arange(
            int(self.config.num_entities),
            device=device,
            dtype=torch.long,
        )
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self._static_candidate_cache = self.encode_static_candidates(entity_ids).detach()
        if was_training:
            self.train()
        return self._static_candidate_cache

    def get_static_candidate_vectors(self, device):
        assert not self.temporal_enabled, (
            "get_static_candidate_vectors returns one time-independent table "
            "and must not be used when temporal_enabled=True; use "
            "build_graph_world_states + _score_with_per_time_candidates instead."
        )
        if self.cache_static_candidates:
            if (
                self._static_candidate_cache is None
                or self._static_candidate_cache.device != device
            ):
                return self.refresh_static_candidate_cache(device=device)
            return self._static_candidate_cache
        entity_ids = torch.arange(
            int(self.config.num_entities),
            device=device,
            dtype=torch.long,
        )
        return self.encode_static_candidates(entity_ids)

    def compute_loss(
        self,
        query_vectors,
        target_vectors,
        truth_mask=None,
    ):
        scores = torch.mm(query_vectors, target_vectors.t()) / self.temperature
        loss = self._filtered_in_batch_contrastive_loss(
            scores,
            truth_mask=truth_mask,
        ).mean()
        return loss, scores

    def score_all_entities(self, h_batch, r_batch, context_batch, graph_state_by_time=None):
        query_vectors, relation_vectors = self.encode_query(
            h_batch, r_batch, context_batch, graph_state_by_time=graph_state_by_time,
        )

        if self.temporal_enabled and graph_state_by_time:
            query_time_ids = h_batch.get('time_id')
            return self._score_with_per_time_candidates(
                query_vectors,
                relation_vectors,
                query_time_ids,
                graph_state_by_time,
            )

        candidate_vectors = self.get_static_candidate_vectors(query_vectors.device)
        if self.decoder_name == 'convtranse':
            return self.decoder(query_vectors, relation_vectors, candidate_vectors)
        return torch.mm(query_vectors, candidate_vectors.t()) / self.temperature

    def _score_with_per_time_candidates(self, query_vectors, relation_vectors, query_time_ids, graph_state_by_time):
        """Scores queries against per-time-id candidate tables and reassembles one score tensor."""
        if query_time_ids is None:
            raise ValueError("Per-time candidate scoring requires query time_ids.")
        query_time_ids = query_time_ids.reshape(-1)
        num_entities = int(self.config.num_entities)
        entity_ids = torch.arange(num_entities, device=query_vectors.device, dtype=torch.long)
        scores = torch.empty(
            query_vectors.size(0), num_entities,
            device=query_vectors.device, dtype=query_vectors.dtype,
        )
        for t in torch.unique(query_time_ids).tolist():
            row_mask = query_time_ids == t
            candidate_vectors = self.encode_target(
                {'id': entity_ids, 'time_id': torch.full_like(entity_ids, int(t))},
                graph_state_by_time=graph_state_by_time,
            )
            if self.decoder_name == 'convtranse':
                scores[row_mask] = self.decoder(
                    query_vectors[row_mask], relation_vectors[row_mask], candidate_vectors,
                )
            else:
                scores[row_mask] = torch.mm(query_vectors[row_mask], candidate_vectors.t()) / self.temperature
        return scores

    @staticmethod
    def compute_full_softmax_loss(scores, target_ids):
        return F.cross_entropy(scores, target_ids)
