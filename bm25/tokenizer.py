import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Dict

from tqdm import tqdm
from transformers import AutoTokenizer


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


STOPWORDS_EN = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "couldn",
    "couldn't",
    "d",
    "did",
    "didn",
    "didn't",
    "do",
    "does",
    "doesn",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "hadn't",
    "has",
    "hasn",
    "hasn't",
    "have",
    "haven",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightn't",
    "more",
    "most",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "needn",
    "needn't",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "y",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.vocab
    print(len(vocab.values()))
    
    # with open("data\\nq_doc.tsv", "r", encoding="utf-8") as f:
    #     for line in f:
    #         cid, text = line.strip().split('\t')
    #         print(tokenizer.encode(text))
    #         break