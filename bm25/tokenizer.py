import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import pickle
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Dict

from tqdm import tqdm
from transformers import AutoTokenizer

from __init__ import STOPWORDS_EN

@dataclass
class TokenizerOutput:
    ids: List[List[int]]
    vocab: Dict[str, int]


class Tokenizer:
    def __init__(self, 
                 tokenizer_path: str = None,
                 token_pattern: str = r"(?u)\b\w\w+\b",
                 stemmer = None,
                 ):
        self.token_pattern = re.compile(token_pattern)
        self.stopwords = set(STOPWORDS_EN)

        # Set stemmer
        if stemmer is not None:
            if callable(stemmer):
                self.stemmer_fn = stemmer
            elif hasattr(stemmer, 'stemWords'):
                self.stemmer_fn = stemmer.stemWords
            else:
                raise ValueError("Unknown stemmer type")
        else:
            self.stemmer_fn = None

        # Load vocab
        self.vocab2ids = None
        if tokenizer_path is not None:
            with open(tokenizer_path, 'r') as f:
                self.vocab2ids = json.load(f)

    def train(self, 
              texts: List[str],
              save_dir: str,
              lowercase: bool = True):
        os.makedirs(save_dir, exist_ok=True)
        token_to_ids = {}
        for text in texts:
            doc_token_ids = []
            if lowercase:
                text = text.lower()
            text_split = self.token_pattern.findall(text)
            for token in text_split:
                if token in self.stopwords:
                    continue
                if token not in token_to_ids:
                    token_to_ids[token] = len(token_to_ids)
                doc_token_ids.append(token_to_ids[token])

        token_list = list(token_to_ids.keys())
        if self.stemmer_fn is not None:
            tokens_stemmed = self.stemmer_fn(token_list)
            vocab = set(tokens_stemmed)
            vocab2ids = {token: i for i, token in enumerate(vocab)}
        else:
            vocab2ids = token_to_ids
        
        vocab2ids["<unk>"] = len(vocab2ids)
        with open(os.path.join(save_dir, 'vocab.json'), 'w') as f:
            json.dump(vocab2ids, f)

        self.vocab2ids = vocab2ids
        

    def encode(self, 
               texts: Union[str, List[str]],
               lowercase: bool = True) -> List:
        assert self.vocab2ids is not None, "Tokenizer must be trained first"

        text_ids = []
        if isinstance(texts, str):
            texts = [texts]
        for text in tqdm(texts, desc="Tokenizing"):
            doc_token_ids = []
            if lowercase:
                text = text.lower()
            text_split = self.token_pattern.findall(text)
            for token in text_split:
                if token in self.stopwords:
                    continue
                if self.stemmer_fn is not None:
                    token = ''.join(self.stemmer_fn(token))
                if token not in self.vocab2ids:
                    doc_token_ids.append(self.vocab2ids["<unk>"])
                else:
                    doc_token_ids.append(self.vocab2ids[token])
            text_ids.append(doc_token_ids)

        return text_ids

    def get_vocab_ids(self):
        return list(self.vocab2ids.values())


def tokenize(args):
    print("Runing tokenization")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    with open(args.corpus_path, "r") as fi, open(args.output_path, "w") as fo:
        lines = fi.readlines()
        total_lines = len(lines)
        print("Total lines: {}".format(total_lines))

        for line in tqdm(lines, total=total_lines):
            cid, text = line.strip().split('\t')
            tokens = tokenizer.encode(text, add_special_tokens=False)
            encoded = {
                "text_id": cid,
                "text": tokens
            }
            fo.write(json.dumps(encoded) + "\n")

def tokenizer_surgery(tokenizer: AutoTokenizer, 
                      save_dir: str = "tokenizer_surgery"):
    unused_tokens = [ token for token in tokenizer.vocab.keys() if 
                     'unused' in token or
                     token in ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]' ] ]
    
    # for token in unused_tokens:
    #     token_id = tokenizer.vocab.pop(token)
    #     tokenizer.ids_to_tokens.pop(token_id)

    unused_tokens = [token for token in tokenizer.get_vocab().keys() if 'unused' in token]
    remaining_tokens = [token for token in tokenizer.get_vocab().keys() if token not in unused_tokens]


    tokenizer = tokenizer.train_new_from_iterator((remaining_tokens), vocab_size=len(remaining_tokens))
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--corpus_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--tokenizer_path", type=str, required=False)

    args = parser.parse_args()
    # tokenize(args)

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # vocab = tokenizer.vocab
    # print(len(vocab.values()))
    
    # with open("data\\nq_doc.tsv", "r", encoding="utf-8") as f:
    #     for line in f:
    #         cid, text = line.strip().split('\t')
    #         print(tokenizer.encode(text))
    #         break

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer_surgery(tokenizer, save_dir=args.tokenizer_path)