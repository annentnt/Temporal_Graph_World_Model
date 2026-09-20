import os
import math
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import json
from torch.optim.lr_scheduler import LambdaLR

# Ensure the project root is importable.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.dataset import (
    CollateFN,
    GWMDataset,
    PackedSnapshotBatchSampler,
    SnapshotEdgeBank,
    TemporalGWMDataset,
    TrainTruthIndex,
)
from utils.seed import make_torch_generator, make_worker_init_fn, seed_everything
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)
from utils.early_stopping import EarlyStopping

def _to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return str(value)


def _get_model_parameter_info(model):
    """Summarizes parameter counts and shapes for the training config log."""
    total_params = 0
    trainable_params = 0
    param_details = []

    for name, param in model.named_parameters():
        numel = int(param.numel())
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

        param_details.append({
            'name': name,
            'shape': list(param.shape),
            'numel': numel,
            'requires_grad': bool(param.requires_grad),
            'dtype': str(param.dtype).replace('torch.', ''),
        })

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'parameters': param_details,
    }


def save_training_config(config, output_dir, args=None, model=None):
    """Writes the effective training config to output_dir."""
    config_dict = {k: _to_serializable(v) for k, v in vars(config).items()}
    if args is not None:
        config_dict['cli_args'] = {k: _to_serializable(v) for k, v in vars(args).items()}
    if model is not None:
        config_dict['model_parameters'] = _get_model_parameter_info(model)

    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)


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

def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mrr, early_stopping):
    """Saves model/optimizer/scheduler state and training progress."""
    torch.save(
        {
            'architecture': (
                'temporal_early_fusion_retemp_v1'
                if bool(getattr(model.config, 'temporal_enabled', False))
                else 'early_fusion_v1'
            ),
            'training_objective': getattr(
                model.config,
                'training_objective',
                'single_positive_unfiltered_in_batch',
            ),
            'epoch': epoch,
            'best_mrr': best_mrr,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_state': {
                'best_value': early_stopping.best_value,
                'counter': early_stopping.counter,
                'should_stop': early_stopping.should_stop,
            },
        },
        path,
    )

def train(args):
    """Runs the full training loop for one config."""
    config = get_config(args)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    seed = int(getattr(config, 'seed', 42))
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading data from {config.data_dir}...")
    temporal_enabled = bool(getattr(config, 'temporal_enabled', False))
    history_len = int(getattr(config, 'history_len', 3))
    if temporal_enabled:
        train_dataset = TemporalGWMDataset(
            config.data_dir,
            split='train',
            context_k=int(getattr(config, 'temporal_context_k', getattr(config, 'context_k', 10))),
            history_len=history_len,
        )
        train_truth_index = TrainTruthIndex(train_dataset.quadruples)
        train_edge_bank = SnapshotEdgeBank(train_dataset.quadruples)
    else:
        train_dataset = GWMDataset(config.data_dir, split='train')
        train_truth_index = TrainTruthIndex(train_dataset.triples)
        train_edge_bank = None

    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        num_ent = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        num_rel = len(json.load(f))

    config.num_entities = num_ent
    config.num_relations = num_rel

    print("Initializing model...")
    model = GWM(config).to(device)

    collate_fn = CollateFN()

    if temporal_enabled:
        # Batches whole time-snapshots together for the graph evolver.
        train_batch_sampler = PackedSnapshotBatchSampler(
            train_dataset, max_batch_size=int(config.batch_size), seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=(device.type == 'cuda'),
            worker_init_fn=make_worker_init_fn(seed),
        )
    else:
        train_batch_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=(device.type == 'cuda'),
            drop_last=True,
            generator=make_torch_generator(seed),
            worker_init_fn=make_worker_init_fn(seed),
        )

    entity_emb_path = os.path.join(config.data_dir, 'entity_text_embeddings.pt')
    relation_emb_path = os.path.join(config.data_dir, 'relation_text_embeddings.pt')
    if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
        raise FileNotFoundError(
            "Missing precomputed text embedding cache files. "
            "Expected entity_text_embeddings.pt and relation_text_embeddings.pt in data_dir."
        )

    model.load_embeddings(
        entity_source=entity_emb_path,
        relation_source=relation_emb_path,
        kind='text',
        freeze=True,
    )
    print("Loaded text embeddings into text embedding tables...")

    structural_entity_source = os.path.join(config.data_dir, 'structural_entities.pt')
    structural_relation_source = os.path.join(config.data_dir, 'structural_relations.pt')
    if not os.path.exists(structural_entity_source) or not os.path.exists(structural_relation_source):
        raise FileNotFoundError(
            "Missing precomputed structural prior files. Expected "
            "structural_entities.pt and structural_relations.pt in data_dir."
        )
    model.load_embeddings(
        entity_source=structural_entity_source,
        relation_source=structural_relation_source,
        kind='structural',
        freeze=False,
    )
    print("Loaded structural embeddings into structural embedding tables...")

    if getattr(model, 'decoder_name', 'dot') == 'convtranse':
        config.training_objective = 'full_entity_convtranse_cross_entropy'
    else:
        config.training_objective = 'single_positive_filtered_in_batch'

    save_training_config(config, config.output_dir, args=args, model=model)

    base_lr = float(config.learning_rate)
    weight_decay = float(getattr(config, 'weight_decay', 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    total_steps = max(1, config.num_epochs * len(train_loader))
    warmup_ratio = float(getattr(config, 'warmup_ratio', 0.0))
    warmup_steps = min(int(total_steps * warmup_ratio), total_steps)
    min_lr = float(getattr(config, 'min_lr', 0.0))
    min_lr_ratio = 0.0 if base_lr <= 0 else max(min_lr / base_lr, 0.0)

    def lr_lambda(step_index):
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(max(1, warmup_steps))

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(max((step_index - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    grad_clip_norm = float(getattr(config, 'grad_clip_norm', 1.0))

    valid_edge_bank = None
    valid_name = 'valid_quadruples.pt' if temporal_enabled else 'valid_triples.pt'
    if os.path.exists(os.path.join(config.data_dir, valid_name)):
        print("Loading validation data...")
        if temporal_enabled:
            # Includes valid-so-far so recent context is available for valid queries.
            valid_quads_path = os.path.join(config.data_dir, 'valid_quadruples.pt')
            valid_history_pool = torch.cat(
                [train_dataset.quadruples, torch.load(valid_quads_path, map_location='cpu').long()],
                dim=0,
            )
            valid_dataset = TemporalGWMDataset(
                config.data_dir,
                split='valid',
                context_k=int(getattr(config, 'temporal_context_k', getattr(config, 'context_k', 10))),
                history_len=history_len,
                history_quadruples=valid_history_pool,
            )
            valid_edge_bank = SnapshotEdgeBank(valid_history_pool)
        else:
            valid_dataset = GWMDataset(config.data_dir, split='valid')
        if temporal_enabled:
            valid_batch_sampler = PackedSnapshotBatchSampler(
                valid_dataset, max_batch_size=int(config.batch_size), seed=seed,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_sampler=valid_batch_sampler,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=(device.type == 'cuda'),
                worker_init_fn=make_worker_init_fn(seed),
            )
        else:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=(device.type == 'cuda'),
                drop_last=False,
                worker_init_fn=make_worker_init_fn(seed),
            )
    else:
        valid_loader = None

    hr_map = None
    all_entity_embeddings = None
    entity_loader = None
    if valid_loader is not None:
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file='ground_truth.json',
            fallback_splits=['train', 'valid', 'test']
        )

        candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(int(config.batch_size), 256)))
        entity_loader = build_entity_loader(
            data_dir=config.data_dir,
            batch_size=candidate_batch_size,
            num_workers=2,
        )

    print("Starting training...")
    train_start_time = time.perf_counter()
    best_mrr = float('-inf')

    early_stopping = EarlyStopping(
        patience=getattr(config, 'early_stopping_patience', getattr(config, 'early_stopping', 10)),
        mode='max'
    )

    log_path = os.path.join(config.output_dir, 'training_log.json')
    history = []

    start_epoch = 0
    resume_path = getattr(args, 'checkpoint', None)
    if getattr(args, 'resume', False):
        if resume_path is None:
            resume_path = os.path.join(config.output_dir, 'best_checkpoint.pt')
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = int(ckpt['epoch']) + 1
            best_mrr = float(ckpt.get('best_mrr', float('-inf')))
            es_state = ckpt.get('early_stopping_state', {})
            early_stopping.best_value = float(es_state.get('best_value', float('-inf')))
            early_stopping.counter = int(es_state.get('counter', 0))
            early_stopping.should_stop = bool(es_state.get('should_stop', False))
            if os.path.exists(log_path):
                with open(log_path) as _f:
                    _existing = json.load(_f)
                history = [e for e in _existing if isinstance(e, dict) and 'event' not in e]
            print(f"  Resumed: start_epoch={start_epoch} | best_mrr={best_mrr:.4f} | es_counter={early_stopping.counter}")
        else:
            print(f"WARNING: --resume set but checkpoint not found at {resume_path}. Starting from scratch.")

    use_full_entity_decoder = getattr(model, 'decoder_name', 'dot') == 'convtranse'

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.perf_counter()
        _sync_device(device)
        model.train()
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        if use_full_entity_decoder and getattr(model, 'cache_static_candidates', False):
            model.refresh_static_candidate_cache(device=device)
        total_loss = 0
        total_filtered_truth_count = 0
        total_query_rows = 0
        filtered_query_rows = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            if use_full_entity_decoder:
                truth_mask = None
                total_query_rows += int(batch['t_batch']['id'].numel())
            else:
                truth_mask = train_truth_index.build_in_batch_truth_mask(
                    head_ids=batch['h_batch']['id'],
                    relation_ids=batch['r_batch']['id'],
                    candidate_tail_ids=batch['t_batch']['id'],
                    time_ids=batch['time_batch']['id'] if batch.get('time_batch') is not None else None,
                    device=device,
                )
                filtered_truths_per_query = truth_mask.sum(dim=1) - 1
                total_filtered_truth_count += int(
                    filtered_truths_per_query.sum().item()
                )
                total_query_rows += int(filtered_truths_per_query.numel())
                filtered_query_rows += int(
                    (filtered_truths_per_query > 0).sum().item()
                )

            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            t_batch = {k: v.to(device) for k, v in batch['t_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

            optimizer.zero_grad()

            graph_state_by_time = None
            if temporal_enabled and getattr(model, 'graph_evolver', None) is not None:
                required_time_ids = [h_batch['time_id']]
                ctx_time_ids = context_batch.get('time_id')
                if ctx_time_ids is not None and ctx_time_ids.numel() > 0:
                    required_time_ids.append(ctx_time_ids)
                graph_state_by_time = model.build_graph_world_states(
                    time_ids=torch.cat(required_time_ids),
                    edge_bank=train_edge_bank,
                    history_len=history_len,
                )

            if use_full_entity_decoder:
                scores = model.score_all_entities(
                    h_batch, r_batch, context_batch, graph_state_by_time=graph_state_by_time,
                )
                loss = model.compute_full_softmax_loss(scores, t_batch['id'])
            else:
                query_vector = model(h_batch, r_batch, context_batch, graph_state_by_time=graph_state_by_time)
                t_fused = model.encode_target(t_batch, graph_state_by_time=graph_state_by_time)
                loss, _ = model.compute_loss(
                    query_vector,
                    t_fused,
                    truth_mask=truth_mask,
                )

            if not torch.isfinite(loss):
                print("Warning: non-finite loss detected; skipping batch to avoid corrupting model weights.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        _sync_device(device)
        epoch_train_seconds = time.perf_counter() - epoch_start_time

        avg_train_loss = total_loss / len(train_loader)
        avg_filtered_truths_per_query = (
            total_filtered_truth_count / max(total_query_rows, 1)
        )
        filtered_query_rate = (
            filtered_query_rows / max(total_query_rows, 1)
        )

        if use_full_entity_decoder:
            print(
                f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | "
                f"Objective: Full-Entity ConvTransE CE | "
                f"Candidates: {config.num_entities} | "
                f"Train Time: {epoch_train_seconds:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | "
                f"Filtered Truths/Query: {avg_filtered_truths_per_query:.4f} | "
                f"Rows with Filtered Truths: {filtered_query_rate:.4f} | "
                f"Train Time: {epoch_train_seconds:.2f}s"
            )

        eval_every = getattr(config, 'eval_every', 1)
        if valid_loader and (epoch + 1) % eval_every == 0:
            model.eval()
            if use_full_entity_decoder and getattr(model, 'cache_static_candidates', False):
                model.refresh_static_candidate_cache(device=device)

            if temporal_enabled:
                all_entity_embeddings = None
            else:
                all_entity_embeddings = encode_all_entities_as_targets(
                    model=model,
                    entity_loader=entity_loader,
                    device=device,
                )

            val_metrics = compute_filtered_ranking_metrics(
                model=model,
                data_loader=valid_loader,
                all_entity_embeddings=all_entity_embeddings,
                hr_map=hr_map,
                device=device,
                desc="Validation",
                edge_bank=valid_edge_bank if temporal_enabled else None,
                history_len=history_len if temporal_enabled else None,
            )

            val_mrr = val_metrics['MRR']
            val_h1 = val_metrics['Hits@1']
            val_h3 = val_metrics['Hits@3']
            val_h10 = val_metrics['Hits@10']
            val_mr = val_metrics['MR']

            print(
                f"Epoch {epoch+1} Val | "
                f"MRR: {val_mrr:.4f} | MR: {val_mr:.2f} | "
                f"Hits@1: {val_h1:.4f} | Hits@3: {val_h3:.4f} | Hits@10: {val_h10:.4f}"
            )

            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_objective': config.training_objective,
                'val_mrr': val_mrr,
                'val_mr': val_mr,
                'val_hits1': val_h1,
                'val_hits3': val_h3,
                'val_hits10': val_h10
            }
            if not use_full_entity_decoder:
                epoch_log['avg_filtered_truths_per_query'] = avg_filtered_truths_per_query
                epoch_log['filtered_query_rate'] = filtered_query_rate
            history.append(epoch_log)
            with open(log_path, 'w') as f:
                json.dump(history, f, indent=2)

            is_best = val_mrr > best_mrr
            if is_best:
                best_mrr = val_mrr

            should_stop = early_stopping(val_mrr)
            if is_best:
                save_checkpoint(
                    os.path.join(config.output_dir, 'best_checkpoint.pt'),
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_mrr,
                    early_stopping,
                )
            if should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"  Best MRR: {early_stopping.best_value:.4f}")
                print(f"  No improvement for {early_stopping.patience} epochs")
        else:
            should_stop = False
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_objective': config.training_objective,
            }
            if not use_full_entity_decoder:
                epoch_log['avg_filtered_truths_per_query'] = avg_filtered_truths_per_query
                epoch_log['filtered_query_rate'] = filtered_query_rate
            history.append(epoch_log)
            with open(log_path, 'w') as f:
                  json.dump(history, f, indent=2)

        save_checkpoint(
            os.path.join(config.output_dir, 'latest_checkpoint.pt'),
            model,
            optimizer,
            scheduler,
            epoch,
            best_mrr,
            early_stopping,
        )
        if should_stop:
            break

    _sync_device(device)
    total_train_seconds = time.perf_counter() - train_start_time
    print(f"Total training time: {total_train_seconds:.2f}s")
    history.append({
        'event': 'training_complete',
        'total_train_seconds': total_train_seconds,
        'epochs_completed': len(history),
    })
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt file (default: <output_dir>/best_checkpoint.pt)')

    args = parser.parse_args()
    train(args)
