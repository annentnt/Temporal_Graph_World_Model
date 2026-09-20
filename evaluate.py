import os
import torch
from torch.utils.data import DataLoader
import argparse
import yaml
import json
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.dataset import (
    CollateFN,
    GWMDataset,
    PackedSnapshotBatchSampler,
    SnapshotEdgeBank,
    TemporalGWMDataset,
)
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)

def get_config(args):
    """Loads the YAML config and applies CLI overrides."""
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    if args.data_dir: config_dict['data_dir'] = args.data_dir
    if args.output_dir: config_dict['output_dir'] = args.output_dir

    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    return Config(config_dict)

def evaluate(args):
    """Loads a trained checkpoint and computes filtered ranking metrics on the test split."""
    config = get_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    eval_batch_size = int(getattr(config, 'eval_batch_size', min(int(config.batch_size), 128)))
    candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(eval_batch_size * 2, 256)))

    print("Loading model...")
    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        config.num_entities = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        config.num_relations = len(json.load(f))

    model = GWM(config).to(device)

    checkpoint_path = os.path.join(config.output_dir, 'best_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}, trying latest...")
        checkpoint_path = os.path.join(config.output_dir, 'latest_checkpoint.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}."
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            "Unsupported legacy checkpoint. Expected a full training "
            "checkpoint containing 'model_state_dict'."
        )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    model.eval()

    temporal_enabled = bool(getattr(config, 'temporal_enabled', False))
    if temporal_enabled:
        all_entity_embeddings = None
        print("Temporal mode: entity candidates will be encoded per query time.")
    else:
        print("Encoding all entities as targets...")
        entity_loader = build_entity_loader(
            data_dir=config.data_dir,
            batch_size=candidate_batch_size,
            num_workers=4,
        )

        all_entity_embeddings = encode_all_entities_as_targets(
            model=model,
            entity_loader=entity_loader,
            device=device
        )
        print(f"Encoded {all_entity_embeddings.size(0)} entities.")

    split = 'test'
    print(f"Evaluating on {split} set...")
    split_file = f'{split}_quadruples.pt' if temporal_enabled else f'{split}_triples.pt'
    if not os.path.exists(os.path.join(config.data_dir, split_file)):
        print(f"Test triples not found, using 'valid' set.")
        split = 'valid'

    history_len = int(getattr(config, 'history_len', 3))
    test_edge_bank = None
    if temporal_enabled:
        # History pool covers train (+valid, +test-so-far for the test split),
        # with lookups always filtered to strictly-past times.
        train_quads_path = os.path.join(config.data_dir, 'train_quadruples.pt')
        history_pool = [torch.load(train_quads_path, map_location='cpu').long()]
        if split == 'test':
            valid_quads_path = os.path.join(config.data_dir, 'valid_quadruples.pt')
            if os.path.exists(valid_quads_path):
                history_pool.append(torch.load(valid_quads_path, map_location='cpu').long())
        own_split_quads_path = os.path.join(config.data_dir, f'{split}_quadruples.pt')
        history_pool.append(torch.load(own_split_quads_path, map_location='cpu').long())
        history_quadruples = torch.cat(history_pool, dim=0)
        test_edge_bank = SnapshotEdgeBank(history_quadruples)

        test_dataset = TemporalGWMDataset(
            config.data_dir,
            split=split,
            context_k=int(getattr(config, 'temporal_context_k', getattr(config, 'context_k', 10))),
            history_len=history_len,
            history_quadruples=history_quadruples,
        )
    else:
        test_dataset = GWMDataset(config.data_dir, split=split)
    collate_fn = CollateFN()

    if temporal_enabled:
        test_batch_sampler = PackedSnapshotBatchSampler(
            test_dataset, max_batch_size=eval_batch_size,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_batch_sampler,
            collate_fn=collate_fn,
            num_workers=4,
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

    if split == 'test':
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file='ground_truth.json',
            fallback_splits=['train', 'valid']
        )
    else:
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file='ground_truth.json',
            fallback_splits=['train']
        )

    predictions_path = os.path.join(config.output_dir, f'predictions_{split}.jsonl')

    metrics = compute_filtered_ranking_metrics(
        model=model,
        data_loader=test_loader,
        all_entity_embeddings=all_entity_embeddings,
        hr_map=hr_map,
        device=device,
        desc="Evaluating",
        save_predictions_path=predictions_path,
        edge_bank=test_edge_bank,
        history_len=history_len if temporal_enabled else None,
    )

    final_mrr = metrics['MRR']
    final_h1 = metrics['Hits@1']
    final_h3 = metrics['Hits@3']
    final_h10 = metrics['Hits@10']

    print(f"\n--- Evaluation Results ({split}) ---")
    print(f"MRR       : {final_mrr:.4f}")
    print(f"Hits@1    : {final_h1:.4f}")
    print(f"Hits@3    : {final_h3:.4f}")
    print(f"Hits@10   : {final_h10:.4f}")
    print("-------------------------------")

    results = {
        'mrr': final_mrr,
        'hits1': final_h1,
        'hits3': final_h3,
        'hits10': final_h10
    }
    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    args = parser.parse_args()
    evaluate(args)
