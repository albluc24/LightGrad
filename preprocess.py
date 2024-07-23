import argparse
import sys, os
import pathlib
import random
import re
import json
import numpy as np
import itertools
from joblib import load
from tqdm import tqdm
sys.path.append(os.path.dirname(os.getcwd()))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", type=str, help="path to training config")
    parser.add_argument("--export_dir", type=str,
                        help="path to save preprocess result", default='dataset/')
    parser.add_argument("--test_sample_count", type=int, default=200)
    parser.add_argument("--valid_sample_count", type=int, default=200)
    return parser.parse_args()


def main():
    args = get_args()
    args.export_dir = pathlib.Path(args.export_dir)
    export_dir =args.export_dir 
    (export_dir/ 'embs').mkdir(parents=True, exist_ok=True)
    (train_dataset, valid_dataset, test_dataset, phn2id) = preprocess(args)
    with open(export_dir / "train_dataset.json", "w") as f:
        json.dump(train_dataset, f)
    with open(export_dir / "valid_dataset.json", "w") as f:
        json.dump(valid_dataset, f)
    with open(export_dir / "test_dataset.json", "w") as f:
        json.dump(test_dataset, f)
    with open(export_dir / "phn2id.json", "w") as f:
        json.dump(phn2id, f)


def preprocess(args):
    numb=0
    mapping=set()
    from training import loaders
    import training.utils
    from training.unit_collector import unitCollector as Embedder
    config_path = pathlib.Path(args.config_path)
    config=training.utils.load_config(config_path)
    kmeans=load(config['kmeans_file'])
    pca=load(config['pca_file'])
    embedder=Embedder(config)
    if config['inv_training_cutoff']>=0:
        data=loaders.islice(training.utils.get_loader_from_config(config), config['inv_training_cutoff'])
        data=tqdm(data, total=config['inv_training_cutoff'])
    else: data=tqdm(training.utils.get_loader_from_config(config))
    meta_info = []
    for sample in data:
        segments=embedder.load_and_split_file(sample)
        training.utils.cleanup_temporary_audio(sample, [16000, 32000])
        for s,e in segments:
            try:
                result=embedder.embed(s,e)
            except Exception as ex: breakpoint()
            if result.shape[0] <=200: continue
            emb=pca.transform(result)
            if np.isnan(emb).any(): breakpoint()
            units=embedder.collector(result)
            text=[i['text'].split('-')[0] for i in units]
            mapping=mapping|set(text)
            emb_path = args.export_dir / "embs" / f"{numb}.npy"
            np.save(emb_path, emb)
            meta_info.append(
                {
                    "name": str(numb),
                    "emb_path": str(emb_path),
                    "phonemes": text,
                    }
                )
            numb+=1
    random.shuffle(meta_info)
    test_dataset = meta_info[: args.test_sample_count]
    valid_dataset = meta_info[
        args.test_sample_count: args.test_sample_count + args.valid_sample_count
    ]
    train_dataset = meta_info[args.test_sample_count +
                              args.valid_sample_count:]
    mapping=mapping|set(('<bos>','<eos>'))
    mapping=list(mapping)
    mapping=['<blank>']+mapping
    mapping= {s: i + 1 for i, s in enumerate(mapping)}
    return train_dataset, valid_dataset, test_dataset, mapping

if __name__ == "__main__":
    main()
