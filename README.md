# T-GWM: Temporal Graph World Model for Temporal Knowledge Graph Forecasting

T-GWM is a Temporal Knowledge Graph Completion (TKGC) model for the
extrapolation / forecasting setting: given the history of a knowledge
graph up to time *t*, predict missing entities in facts at time *t*.
The model combines a text (frozen BERT) and structural view of each
entity/relation, a diachronic (trend + seasonal) temporal encoder, and a
recurrent graph world-state evolver over historical snapshots.

## Repository layout

```
T-GWM/
├── configs/                  # One YAML per dataset / experiment variant
├── data/                     # Raw ICEWS14 / ICEWS14s / ICEWS18 / ICEWS05-15 / GDELT triples
├── model/
│   ├── model.py               # GWM model (fusion, temporal encoder, decoder)
│   ├── graph_evolver.py        # Graph world-state evolver (snapshot message passing)
│   └── dataset.py             # Datasets, batch sampler, edge bank
├── utils/
│   ├── preprocess_temporal_data.py  # Raw triples -> processed tensors + text embeddings
│   ├── eval.py                 # Filtered ranking metrics
│   ├── seed.py                 # Reproducibility helpers
│   └── early_stopping.py
├── tests/
│   └── test_model_dataset.py   # Unit tests (unittest)
├── train_temporal_structural_priors.py  # Pretrains structural entity/relation priors
├── train.py                    # Main training loop
├── evaluate.py                 # Filtered-ranking evaluation on the test split
├── visualize_training.py       # Plot training_log.json loss/MRR curves
└── requirements.txt
```

## Installation

Requires Python 3.9+ and a CUDA GPU (training/evaluation fall back to CPU
automatically but will be slow for the temporal graph evolver).

```bash
pip install -r requirements.txt
```

## Datasets

Raw triples for five datasets are already included under `data/`:
`ICEWS14`, `ICEWS14s`, `ICEWS18`, `ICEWS05-15`, `GDELT`. Each folder
contains `train.txt` / `valid.txt` / `test.txt` (tab-separated
`head \t relation \t tail \t time_id`), `entity2id.txt`, `relation2id.txt`.

| Dataset | `data/` folder | processed `data_dir` | config |
|---|---|---|---|
| ICEWS14 | `ICEWS14` | `data-processed/icews14` | `configs/icews14-temporal.yaml` |
| ICEWS14s | `ICEWS14s` | `data-processed/icews14s` | `configs/icews14s-temporal.yaml` |
| ICEWS18 | `ICEWS18` | `data-processed/icews18` | `configs/icews18-temporal.yaml` |
| ICEWS05-15 | `ICEWS05-15` | `data-processed/icews05-15` | `configs/icews05-15-temporal.yaml` |
| GDELT | `GDELT` | `data-processed/gdelt` | `configs/gdelt-temporal.yaml` |

## Usage

Each dataset goes through the same three steps. Commands below use
ICEWS14; substitute the `data/`, `data-processed/`, and config paths from
the table above for other datasets.

### 1. Preprocess

Builds `train/valid/test_quadruples.pt`, adds inverse relations for
bidirectional (head + tail prediction) evaluation, and precomputes frozen
BERT text embeddings for every entity and relation.

```bash
python utils/preprocess_temporal_data.py \
    --data_dir data/ICEWS14 \
    --output_dir data-processed/icews14 \
    --pretrained_model bert-base-uncased \
    --text_device cuda
```

### 2. Pretrain structural priors

Runs a CompGCN-style pass over the training snapshots to produce initial
structural entity/relation embeddings, saved as
`data-processed/icews14/structural_entities.pt` /
`structural_relations.pt`.

```bash
python train_temporal_structural_priors.py \
    --processed_dir data-processed/icews14 \
    --dim 200 --layers 1 --epochs 30
```

### 3. Train

```bash
python train.py --config configs/icews14-temporal.yaml
```

Add `--resume` to continue from `<output_dir>/best_checkpoint.pt` if it
exists. `--data_dir` / `--output_dir` override the paths in the config
file.

### 4. Evaluate

```bash
python evaluate.py --config configs/icews14-temporal.yaml
```

Writes filtered MRR / Hits@1 / Hits@3 / Hits@10 to
`<output_dir>/evaluation_results.json` and per-query predictions to
`<output_dir>/predictions_test.jsonl`.

### Visualize training

```bash
python visualize_training.py --log_path output/icews14-temporal/training_log.json --output training_curves.png
```

## Tests

```bash
python -m unittest tests.test_model_dataset -v
```

## Configs

`configs/` includes the main config per dataset (`*-temporal.yaml`) plus
a few ablation variants used in the paper's experiments (e.g.
`icews14-temporal-no-text.yaml` disables the text modality;
`icews14-temporal-dimmatched.yaml` narrows `fusion_dim` to match the
baseline's hidden size). All ablation variants share the same
preprocessing/structural-prior step as their base dataset -- only `train.py`
and `evaluate.py` need to be re-run with the ablation config.
