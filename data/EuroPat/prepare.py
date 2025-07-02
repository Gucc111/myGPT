import argparse
import os
import urllib.request
import zipfile
from array import array
import tiktoken
import numpy as np
import pickle
from typing import Tuple


def download_and_extract(url: str) -> None:
    """
    下载 ZIP 并解压到指定目录。
    """
    dir_path = os.path.dirname(__file__)
    
    zip_name = os.path.basename(url)
    zip_path = os.path.join(dir_path, zip_name)
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name} ...")
        urllib.request.urlretrieve(url, zip_path)
        print('Download successfully.')
    else:
        print(f"{zip_name} already exists, skipping download.")
    
    datadir_name = zip_name.rsplit('.', maxsplit=1)[0]
    datadir_path = os.path.join(dir_path, datadir_name)
    if not os.path.exists(datadir_path):
        print(f"Extracting {zip_name} to {datadir_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(datadir_path)
        print('Extract successfully.')
    else:
        print(f"{datadir_name} already exists, skipping extraction.")


def process_data(dataset: list[str], lang_name: str = 'en', split: str = 'train', encoder_name: str = 'gpt2') -> tuple[int, int]:
    """
    进行 BPE 编码，并追加写入 output_path
    """
    enc = tiktoken.get_encoding(encoder_name)
    # 清理旧文件
    text_path = os.path.join(os.path.dirname(__file__), lang_name + split + '.bin')
    if os.path.exists(text_path):
        os.remove(text_path)
        print('Clean out-dated files.')
    
    offsets = [0]
    total = 0
    with open(text_path, 'ab') as fout:
        buf = array('H')
        print(f'Processing {lang_name}.')
        for line in dataset:
            text = line.strip() + ' '
            ids = enc.encode_ordinary(text)
            buf.fromlist(ids)
            fout.write(buf.tobytes())
            total += len(ids)
            offsets.append(total)
            buf = array('H')
        print(f'Save {lang_name} bin file successfully.')
    # 保存 offsets
    np.save(os.path.join(os.path.dirname(__file__), lang_name + split + '.offsets.npy'), np.array(offsets, dtype=np.int32))
    print(f'Save the {lang_name} offset file successfully.')
    return enc.n_vocab, offsets[-1]


def split_dataset(data_path: str, split_ratio: float = 0.9) -> Tuple[list, list]:
    """
    按句数拆分为训练和验证集。
    """
    with open(data_path, 'r') as f:
        data = f.readlines()
    n: int = len(data)
    i: int = int(n * split_ratio)
    return data[:i], data[i:]


def main() -> None:
    url = 'https://object.pouta.csc.fi/OPUS-EuroPat/v3/moses/de-en.txt.zip'
    
    parser = argparse.ArgumentParser(description="BPE encoding the EuroPat de-en dataset.")
    parser.add_argument('--split-ratio', type=float, default=0.9, help='训练集比例')
    args = parser.parse_args()

    download_and_extract(url)

    zip_name = os.path.basename(url)
    dir_name = zip_name.rsplit('.', maxsplit=1)[0]
    base_path = os.path.dirname(__file__)
    path_list = [os.path.join(base_path, dir_name, name) for name in ('EuroPat.de-en.en', 'EuroPat.de-en.de')]
    lang_type = ['en', 'de']
    meta = dict()

    for p, lang in zip(path_list, lang_type):
        train_set, val_set = split_dataset(p, split_ratio=args.split_ratio)
        vocab_size, train_tokens = process_data(train_set, lang_name=lang, split='train')
        _, val_tokens = process_data(val_set, lang_name=lang, split='val')
        meta[lang + '_vocab_size'] = vocab_size
        print(f'{lang} has {train_tokens} train tokens and {val_tokens} val tokens.')
    
    meta['<bos>'] = vocab_size
    meta['<eos>'] = vocab_size + 1
    meta['<pad>'] = vocab_size + 2
    
    with open(os.path.join(base_path, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)


if __name__ == '__main__':
    main()
