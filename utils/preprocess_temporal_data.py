import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def read_id_dictionary(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, idx = line.strip().split('\t')
            mapping[name] = int(idx)
    return dict(sorted(mapping.items(), key=lambda item: item[1]))


def read_quadruples(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s, r, o, time_id = line.strip().split('\t')
            rows.append((int(s), int(r), int(o), int(time_id)))
    return rows


def add_inverse_quadruples(rows, num_original_relations):
    inverse_rows = [
        (o, r + num_original_relations, s, time_id)
        for s, r, o, time_id in rows
    ]
    return rows + inverse_rows


def build_ground_truth(*quad_tensors):
    ground_truth = {}
    for tensor in quad_tensors:
        for h, r, t, time_id in tensor.tolist():
            key = f"{h},{r},{time_id}"
            ground_truth.setdefault(key, set()).add(t)
    return {key: sorted(value) for key, value in ground_truth.items()}


def clean_label(label):
    return label.replace('_', ' ').replace(',', ', ').strip()


def build_text_maps(entity2id, relation2id, num_original_relations):
    entity_text = {
        str(entity_id): clean_label(entity_name)
        for entity_name, entity_id in entity2id.items()
    }
    relation_text = {}
    id_to_relation = {idx: name for name, idx in relation2id.items()}
    for relation_id in range(len(id_to_relation)):
        relation_name = id_to_relation[relation_id]
        if relation_id >= num_original_relations and relation_name.endswith('_inv'):
            base_name = relation_name[:-4]
            relation_text[str(relation_id)] = f"inverse of {clean_label(base_name)}"
        else:
            relation_text[str(relation_id)] = clean_label(relation_name)
    return entity_text, relation_text


def precompute_text_embeddings(
    text_dict,
    size,
    output_path,
    pretrained_model='bert-base-uncased',
    batch_size=128,
    max_length=64,
    device=None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    encoder = AutoModel.from_pretrained(pretrained_model).to(device)
    encoder.eval()

    texts = [text_dict.get(str(i), f"token {i}") for i in range(size)]
    chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, size, batch_size), desc=f"Encoding {output_path.name}"):
            batch_text = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = encoder(**encoded)
            chunks.append(output.last_hidden_state[:, 0, :].detach().cpu())

    embeddings = torch.cat(chunks, dim=0).contiguous()
    torch.save(
        {
            'embeddings': embeddings,
            'model_name': pretrained_model,
            'embedding_dim': int(embeddings.size(1)),
        },
        output_path,
    )
    return embeddings.size(1)


def process_temporal_dataset(
    data_dir,
    output_dir,
    add_inverse=True,
    add_inverse_to_eval=True,
    encode_text=True,
    pretrained_model='bert-base-uncased',
    text_batch_size=128,
    max_entity_length=64,
    max_relation_length=64,
    text_device=None,
):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entity2id = read_id_dictionary(data_path / 'entity2id.txt')
    relation2id_raw = read_id_dictionary(data_path / 'relation2id.txt')
    num_original_relations = len(relation2id_raw)

    relation2id = dict(relation2id_raw)
    if add_inverse:
        for name, idx in relation2id_raw.items():
            relation2id[f"{name}_inv"] = idx + num_original_relations

    train_rows = read_quadruples(data_path / 'train.txt')
    valid_rows = read_quadruples(data_path / 'valid.txt')
    test_rows = read_quadruples(data_path / 'test.txt')

    if add_inverse:
        train_rows = add_inverse_quadruples(train_rows, num_original_relations)
        if add_inverse_to_eval:
            valid_rows = add_inverse_quadruples(valid_rows, num_original_relations)
            test_rows = add_inverse_quadruples(test_rows, num_original_relations)

    train_tensor = torch.tensor(train_rows, dtype=torch.long)
    valid_tensor = torch.tensor(valid_rows, dtype=torch.long)
    test_tensor = torch.tensor(test_rows, dtype=torch.long)

    with open(out_path / 'entity2id.json', 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, indent=2)
    with open(out_path / 'relation2id.json', 'w', encoding='utf-8') as f:
        json.dump(relation2id, f, indent=2)

    entity_text, relation_text = build_text_maps(
        entity2id=entity2id,
        relation2id=relation2id,
        num_original_relations=num_original_relations,
    )
    with open(out_path / 'entity_text.json', 'w', encoding='utf-8') as f:
        json.dump(entity_text, f, indent=2)
    with open(out_path / 'relation_text.json', 'w', encoding='utf-8') as f:
        json.dump(relation_text, f, indent=2)

    all_times = sorted(
        set(train_tensor[:, 3].tolist())
        | set(valid_tensor[:, 3].tolist())
        | set(test_tensor[:, 3].tolist())
    )
    time2id = {str(time_id): int(time_id) for time_id in all_times}
    with open(out_path / 'time2id.json', 'w', encoding='utf-8') as f:
        json.dump(time2id, f, indent=2)

    torch.save(train_tensor, out_path / 'train_quadruples.pt')
    torch.save(valid_tensor, out_path / 'valid_quadruples.pt')
    torch.save(test_tensor, out_path / 'test_quadruples.pt')

    ground_truth = build_ground_truth(train_tensor, valid_tensor, test_tensor)
    with open(out_path / 'ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f)

    metadata = {
        'num_entities': len(entity2id),
        'num_original_relations': num_original_relations,
        'num_relations': len(relation2id),
        'num_times': len(all_times),
        'min_time_id': min(all_times),
        'max_time_id': max(all_times),
        'add_inverse': bool(add_inverse),
        'add_inverse_to_eval': bool(add_inverse_to_eval),
    }
    with open(out_path / 'temporal_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    if encode_text:
        entity_dim = precompute_text_embeddings(
            text_dict=entity_text,
            size=len(entity2id),
            output_path=out_path / 'entity_text_embeddings.pt',
            pretrained_model=pretrained_model,
            batch_size=text_batch_size,
            max_length=max_entity_length,
            device=text_device,
        )
        relation_dim = precompute_text_embeddings(
            text_dict=relation_text,
            size=len(relation2id),
            output_path=out_path / 'relation_text_embeddings.pt',
            pretrained_model=pretrained_model,
            batch_size=text_batch_size,
            max_length=max_relation_length,
            device=text_device,
        )
        metadata['text_embedding_model'] = pretrained_model
        metadata['text_entity_embedding_dim'] = int(entity_dim)
        metadata['text_relation_embedding_dim'] = int(relation_dim)
        with open(out_path / 'temporal_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved temporal processed data to {out_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--no_inverse', action='store_true')
    parser.add_argument('--no_inverse_eval', action='store_true')
    parser.add_argument('--no_encode_text', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--text_batch_size', type=int, default=128)
    parser.add_argument('--max_entity_length', type=int, default=64)
    parser.add_argument('--max_relation_length', type=int, default=64)
    parser.add_argument('--text_device', type=str, default=None)
    args = parser.parse_args()

    process_temporal_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        add_inverse=not args.no_inverse,
        add_inverse_to_eval=not args.no_inverse_eval,
        encode_text=not args.no_encode_text,
        pretrained_model=args.pretrained_model,
        text_batch_size=args.text_batch_size,
        max_entity_length=args.max_entity_length,
        max_relation_length=args.max_relation_length,
        text_device=args.text_device,
    )
