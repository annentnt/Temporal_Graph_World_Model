import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.dataset import (
    CollateFN,
    GWMDataset,
    PackedSnapshotBatchSampler,
    SnapshotEdgeBank,
    TrainTruthIndex,
)
from model.graph_evolver import GraphWorldStateEvolver, compute_related_relation_embedding
from model.model import ContextAggregator, GWM


def make_config(context_agg='mean'):
    return SimpleNamespace(
        num_entities=4,
        num_relations=4,
        text_emb_dim=6,
        struct_emb_dim=4,
        fusion_dim=5,
        text_adapter_dim=6,
        struct_adapter_dim=4,
        dynamics_layers=1,
        dropout=0.0,
        adapter_dropout=0.0,
        temperature=0.1,
        context_agg=context_agg,
    )


def make_temporal_config(context_agg='mean'):
    config = make_config(context_agg)
    config.temporal_enabled = True
    return config


class ModelTests(unittest.TestCase):
    def test_structural_freeze_targets_structural_tables(self):
        model = GWM(make_config())
        model.load_embeddings(
            torch.randn(4, 4),
            torch.randn(4, 4),
            kind='structural',
            freeze=True,
        )
        self.assertFalse(model.struct_ent_embs.weight.requires_grad)
        self.assertFalse(model.struct_rel_embs.weight.requires_grad)
        self.assertTrue(model.text_ent_embs.weight.requires_grad)

    def test_isolated_head_preserves_self_state(self):
        for reduction in ('mean', 'max'):
            with self.subTest(reduction=reduction):
                layer = ContextAggregator(
                    hidden_dim=4,
                    reduction=reduction,
                )
                output = layer(
                    head_feat=torch.randn(2, 4),
                    nbr_entity_feat=torch.empty(0, 4),
                    nbr_relation_feat=torch.empty(0, 4),
                    nbr_batch_index=torch.empty(0, dtype=torch.long),
                )
                self.assertEqual(output.shape, (2, 4))
                self.assertTrue(torch.isfinite(output).all())
                self.assertFalse(torch.equal(output, torch.zeros_like(output)))

    def test_context_aggregator_mean_and_max_reductions(self):
        messages = torch.tensor(
            [[1.0, 3.0], [5.0, 2.0], [7.0, 9.0]]
        )
        batch_index = torch.tensor([0, 0, 1])
        reference = torch.zeros(2, 2)

        mean_layer = ContextAggregator(2, reduction='mean')
        max_layer = ContextAggregator(2, reduction='max')

        mean_result = mean_layer._aggregate(
            messages, batch_index, batch_size=2, reference=reference
        )
        max_result = max_layer._aggregate(
            messages, batch_index, batch_size=2, reference=reference
        )

        self.assertTrue(
            torch.equal(mean_result, torch.tensor([[3.0, 2.5], [7.0, 9.0]]))
        )
        self.assertTrue(
            torch.equal(max_result, torch.tensor([[5.0, 3.0], [7.0, 9.0]]))
        )

    def test_context_aggregator_forward_is_residual_pooling_then_norm(self):
        head = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        neighbor_entities = torch.tensor(
            [[2.0, 1.0], [4.0, 2.0], [1.0, 3.0]]
        )
        neighbor_relations = torch.tensor(
            [[1.0, 2.0], [0.5, 3.0], [2.0, 1.0]]
        )
        batch_index = torch.tensor([0, 0, 1])

        layer = ContextAggregator(2, reduction='mean')
        output = layer(
            head,
            neighbor_entities,
            neighbor_relations,
            batch_index,
        )

        composed = neighbor_entities * neighbor_relations
        pooled = torch.stack(
            [composed[:2].mean(dim=0), composed[2]]
        )
        expected = torch.nn.functional.layer_norm(
            head + pooled,
            normalized_shape=(2,),
        )
        self.assertTrue(torch.allclose(output, expected))

    def test_query_aware_truth_mask_finds_distinct_valid_tails(self):
        truth_index = TrainTruthIndex(
            torch.tensor(
                [
                    [0, 0, 2],
                    [0, 0, 3],
                    [1, 0, 4],
                ]
            )
        )
        mask = truth_index.build_in_batch_truth_mask(
            head_ids=torch.tensor([0, 0, 1]),
            relation_ids=torch.tensor([0, 0, 0]),
            candidate_tail_ids=torch.tensor([2, 3, 4]),
        )
        expected = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_query_aware_truth_mask_does_not_use_unseen_answers(self):
        truth_index = TrainTruthIndex(
            torch.tensor(
                [
                    [0, 0, 2],
                    [1, 0, 5],
                ]
            )
        )
        mask = truth_index.build_in_batch_truth_mask(
            head_ids=torch.tensor([0, 1]),
            relation_ids=torch.tensor([0, 0]),
            candidate_tail_ids=torch.tensor([2, 5]),
        )
        self.assertFalse(mask[0, 1].item())
        self.assertFalse(mask[1, 0].item())

    def test_filtered_loss_ignores_other_training_truths(self):
        scores = torch.tensor(
            [
                [2.0, 20.0, 0.0],
                [20.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        truth_mask = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ]
        )
        losses = GWM._filtered_in_batch_contrastive_loss(
            scores,
            truth_mask=truth_mask,
        )
        expected = torch.tensor(
            [
                torch.log1p(torch.exp(torch.tensor(-2.0))),
                torch.log1p(torch.exp(torch.tensor(-2.0))),
                torch.log1p(2 * torch.exp(torch.tensor(-2.0))),
            ]
        )
        self.assertTrue(torch.allclose(losses, expected))

    def test_filtered_loss_penalizes_unrelated_high_score(self):
        scores = torch.tensor([[2.0, 20.0], [0.0, 2.0]])
        truth_mask = torch.eye(2, dtype=torch.bool)
        losses = GWM._filtered_in_batch_contrastive_loss(
            scores,
            truth_mask=truth_mask,
        )
        self.assertGreater(losses[0].item(), 17.0)

    def test_early_fusion_loss_backpropagates_to_gate(self):
        for reduction in ('mean', 'max'):
            with self.subTest(reduction=reduction):
                model = GWM(make_config(reduction))
                h_batch = {'id': torch.tensor([0, 1])}
                r_batch = {'id': torch.tensor([0, 1])}
                t_batch = {'id': torch.tensor([2, 3])}
                context_batch = {
                    'id': torch.tensor([1, 2]),
                    'rel_id': torch.tensor([0, 1]),
                    'batch_index': torch.tensor([0, 1]),
                }
                query = model(h_batch, r_batch, context_batch)
                targets = model.encode_target(t_batch)
                loss, scores = model.compute_loss(
                    query,
                    targets,
                    truth_mask=torch.eye(2, dtype=torch.bool),
                )
                self.assertEqual(scores.shape, (2, 2))
                self.assertEqual(query.shape, (2, 5))
                self.assertEqual(targets.shape, (2, 5))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.entity_fusion.gate[1].weight.grad)
                self.assertIsNotNone(model.relation_fusion.gate[1].weight.grad)

    def test_disable_text_modality_bypasses_text_branch_and_gate(self):
        config = make_config()
        config.disable_text_modality = True
        model = GWM(config)

        entity_ids = torch.tensor([0, 1])
        relation_ids = torch.tensor([0, 1])

        ent_fused_before = model._encode_entity(entity_ids)
        rel_fused_before = model._encode_relation(relation_ids)
        with torch.no_grad():
            model.text_ent_embs.weight.add_(torch.randn_like(model.text_ent_embs.weight))
            model.text_rel_embs.weight.add_(torch.randn_like(model.text_rel_embs.weight))
        ent_fused_after = model._encode_entity(entity_ids)
        rel_fused_after = model._encode_relation(relation_ids)

        self.assertTrue(torch.allclose(ent_fused_before, ent_fused_after))
        self.assertTrue(torch.allclose(rel_fused_before, rel_fused_after))

        ent_fused_after.sum().backward()
        self.assertIsNone(model.entity_fusion.gate[1].weight.grad)
        self.assertIsNone(model.text_adapter.fc1.weight.grad)

    def test_encode_entity_uses_graph_state_when_provided(self):
        model = GWM(make_temporal_config())
        entity_ids = torch.tensor([0, 1])
        time_ids = torch.tensor([5, 5])

        baseline = model._encode_entity(entity_ids, time_ids)

        custom_state = torch.randn(model.config.num_entities, model.struct_emb_dim)
        with_graph_state = model._encode_entity(
            entity_ids, time_ids, graph_state_by_time={5: custom_state},
        )

        self.assertEqual(with_graph_state.shape, baseline.shape)
        self.assertFalse(torch.allclose(with_graph_state, baseline))

    def test_score_all_entities_uses_per_time_candidates_when_temporal(self):
        model = GWM(make_temporal_config())
        model.decoder_name = 'dot'
        h_batch = {'id': torch.tensor([0, 1]), 'time_id': torch.tensor([5, 6])}
        r_batch = {'id': torch.tensor([0, 1])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'batch_index': torch.tensor([0, 1]),
            'time_id': torch.tensor([5, 6]),
        }
        graph_state_by_time = {
            5: torch.randn(model.config.num_entities, model.struct_emb_dim),
            6: torch.randn(model.config.num_entities, model.struct_emb_dim),
        }
        scores = model.score_all_entities(
            h_batch, r_batch, context_batch, graph_state_by_time=graph_state_by_time,
        )
        self.assertEqual(scores.shape, (2, model.config.num_entities))
        self.assertTrue(torch.isfinite(scores).all())


class SnapshotEdgeBankTests(unittest.TestCase):
    def test_get_exact_snapshot_edges_returns_only_same_day_edges(self):
        quadruples = torch.tensor(
            [
                [0, 0, 1, 0],
                [1, 0, 2, 1],
                [2, 1, 3, 1],
                [3, 0, 0, 2],
            ]
        )
        bank = SnapshotEdgeBank(quadruples)
        same_day = bank.get_exact_snapshot_edges(1)
        self.assertEqual(same_day.shape, (2, 3))
        self.assertEqual(sorted(same_day[:, 1].tolist()), [0, 1])

    def test_get_exact_snapshot_edges_empty_for_unknown_time(self):
        quadruples = torch.tensor([[0, 0, 1, 0]])
        bank = SnapshotEdgeBank(quadruples)
        edges = bank.get_exact_snapshot_edges(999)
        self.assertEqual(edges.shape, (0, 3))

    def test_window_excludes_current_and_future_times(self):
        quadruples = torch.tensor(
            [
                [0, 0, 1, 0],
                [1, 0, 2, 1],
                [2, 0, 3, 2],
                [3, 0, 0, 3],
            ]
        )
        bank = SnapshotEdgeBank(quadruples)
        window = bank.get_window_edges_by_time(time_id=3, history_len=2)
        self.assertEqual([t for t, _ in window], [1, 2])

    def test_window_defaults_to_all_strictly_past_snapshots(self):
        quadruples = torch.tensor(
            [
                [0, 0, 1, 0],
                [1, 0, 2, 1],
                [2, 0, 3, 5],
            ]
        )
        bank = SnapshotEdgeBank(quadruples)
        window = bank.get_window_edges_by_time(time_id=5, history_len=None)
        self.assertEqual([t for t, _ in window], [0, 1])


class PackedSnapshotBatchSamplerTests(unittest.TestCase):
    def test_batches_stay_within_max_size_and_cover_every_row_once(self):
        rows = []
        for time_id, group_size in ((0, 3), (1, 2), (2, 4)):
            rows.extend([0, 0, 0, time_id] for _ in range(group_size))
        dataset = SimpleNamespace(quadruples=torch.tensor(rows))

        sampler = PackedSnapshotBatchSampler(dataset, max_batch_size=5, seed=0)
        batches = list(sampler)

        all_indices = sorted(idx for batch in batches for idx in batch)
        self.assertEqual(all_indices, list(range(len(rows))))
        for batch in batches:
            self.assertLessEqual(len(batch), 5)

    def test_oversized_group_still_gets_its_own_batch(self):
        rows = [[0, 0, 0, 0] for _ in range(7)]
        dataset = SimpleNamespace(quadruples=torch.tensor(rows))

        sampler = PackedSnapshotBatchSampler(dataset, max_batch_size=5, seed=0)
        batches = list(sampler)

        self.assertEqual(len(batches), 1)
        self.assertEqual(sorted(batches[0]), list(range(7)))


class GraphWorldStateEvolverTests(unittest.TestCase):
    def test_forward_shapes_and_gradient_flow(self):
        num_entities, dim, num_relations = 4, 4, 2
        evolver = GraphWorldStateEvolver(dim=dim, num_layers=1, dropout=0.0)
        base_state = torch.randn(num_entities, dim, requires_grad=True)
        rel_emb_table = torch.randn(num_relations, dim, requires_grad=True)
        edges_t0 = torch.tensor([[0, 0, 1], [1, 1, 2]])
        edges_t1 = torch.tensor([[2, 0, 3]])

        states = evolver(
            base_entity_state=base_state,
            rel_emb_table=rel_emb_table,
            time_ordered_edges=[(0, edges_t0), (1, edges_t1)],
            num_entities=num_entities,
        )

        self.assertEqual(set(states.keys()), {0, 1})
        for state in states.values():
            self.assertEqual(state.shape, (num_entities, dim))
            self.assertTrue(torch.isfinite(state).all())

        states[1].sum().backward()
        self.assertIsNotNone(evolver.layers[0].message_layer.weight.grad)
        self.assertIsNotNone(evolver.gate.score_proj.weight.grad)
        self.assertIsNotNone(evolver.layers[0].isolated_loop_layer.weight.grad)

    def test_same_snapshot_edges_change_composition_and_backprop_to_rel_emb(self):
        num_entities, dim, num_relations = 4, 4, 2
        evolver = GraphWorldStateEvolver(dim=dim, num_layers=1, dropout=0.0)
        base_state = torch.randn(num_entities, dim, requires_grad=True)
        rel_emb_table = torch.randn(num_relations, dim, requires_grad=True)
        edges_t0 = torch.tensor([[0, 0, 1], [1, 1, 2]])
        same_snapshot_edges = torch.tensor([[0, 1, 9], [2, 0, 9]])

        baseline = evolver(
            base_entity_state=base_state,
            rel_emb_table=rel_emb_table,
            time_ordered_edges=[(0, edges_t0)],
            num_entities=num_entities,
        )
        with_signal = evolver(
            base_entity_state=base_state,
            rel_emb_table=rel_emb_table,
            time_ordered_edges=[(0, edges_t0)],
            num_entities=num_entities,
            same_snapshot_edges=same_snapshot_edges,
        )

        self.assertFalse(torch.allclose(baseline[0], with_signal[0]))

        with_signal[0].sum().backward()
        self.assertIsNotNone(rel_emb_table.grad)
        self.assertTrue(torch.isfinite(rel_emb_table.grad).all())


class ComputeRelatedRelationEmbeddingTests(unittest.TestCase):
    def test_averages_relation_embeddings_per_source_entity(self):
        rel_emb_table = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        snapshot_edges = torch.tensor([[0, 0, 99], [0, 1, 99], [2, 0, 99]])
        related = compute_related_relation_embedding(
            snapshot_edges, rel_emb_table, num_entities=4,
        )
        self.assertTrue(torch.allclose(related[0], torch.tensor([0.5, 0.5])))
        self.assertTrue(torch.allclose(related[2], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(related[1], torch.zeros(2)))
        self.assertTrue(torch.allclose(related[3], torch.zeros(2)))

    def test_empty_edges_return_zeros(self):
        rel_emb_table = torch.randn(3, 5)
        related = compute_related_relation_embedding(
            torch.empty(0, 3, dtype=torch.long), rel_emb_table, num_entities=2,
        )
        self.assertTrue(torch.equal(related, torch.zeros(2, 5)))


class BuildGraphWorldStatesTests(unittest.TestCase):
    def test_related_emb_strips_inverse_relations_to_prevent_answer_leak(self):
        config = make_temporal_config()
        config.num_relations = 4
        model = GWM(config)

        quadruples = torch.tensor([
            [0, 1, 1, 9],
            [0, 1, 1, 10],
            [1, 3, 0, 10],
        ])
        edge_bank = SnapshotEdgeBank(quadruples)

        result = model.build_graph_world_states(
            time_ids=torch.tensor([10]), edge_bank=edge_bank, history_len=2,
        )
        self.assertIn(10, result)
        self.assertEqual(result[10].shape, (config.num_entities, model.struct_emb_dim))
        self.assertTrue(torch.isfinite(result[10]).all())

    def test_distant_time_windows_do_not_leak_into_each_other(self):
        model = GWM(make_temporal_config())
        quadruples = torch.tensor(
            [
                [0, 0, 1, 5],
                [1, 0, 2, 6],
                [2, 0, 3, 1000],
                [3, 0, 0, 1001],
            ]
        )
        edge_bank = SnapshotEdgeBank(quadruples)

        combined = model.build_graph_world_states(
            time_ids=torch.tensor([7, 1002]), edge_bank=edge_bank, history_len=3,
        )
        alone_early = model.build_graph_world_states(
            time_ids=torch.tensor([7]), edge_bank=edge_bank, history_len=3,
        )
        alone_late = model.build_graph_world_states(
            time_ids=torch.tensor([1002]), edge_bank=edge_bank, history_len=3,
        )

        self.assertTrue(torch.equal(combined[7], alone_early[7]))
        self.assertTrue(torch.equal(combined[1002], alone_late[1002]))


class DatasetTests(unittest.TestCase):
    def _write_data(self, root):
        root = Path(root)
        (root / 'entity2id.json').write_text(
            json.dumps({'a': 0, 'b': 1, 'c': 2}), encoding='utf-8'
        )
        (root / 'relation2id.json').write_text(
            json.dumps({'r': 0, 'r_inv': 1}), encoding='utf-8'
        )
        torch.save(torch.tensor([[0, 0, 1]]), root / 'train_triples.pt')
        torch.save(
            {
                'entity_ids': torch.tensor([[1, 2], [0, -1], [-1, -1]]),
                'relation_ids': torch.tensor([[0, 0], [1, -1], [-1, -1]]),
                'mask': torch.tensor(
                    [[True, True], [True, False], [False, False]]
                ),
                'pad_value': -1,
            },
            root / 'context_neighbors.pt',
        )

    def test_answer_edge_removed_and_collated_as_ragged_context(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            item = GWMDataset(root, split='train')[0]
            self.assertEqual(item['context_mask'].tolist(), [False, True])
            batch = CollateFN()([item])
            self.assertEqual(batch['context_batch']['id'].tolist(), [2])
            self.assertEqual(
                set(batch['context_batch']),
                {'id', 'rel_id', 'batch_index'},
            )


if __name__ == '__main__':
    unittest.main()
