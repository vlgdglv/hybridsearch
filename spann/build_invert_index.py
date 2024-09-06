import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import argparse
import logging
import numpy as np
from tqdm import tqdm

from invert_index import InvertIndex


def build_invert_index(args):
    index = InvertIndex(args.index_dir, args.index_name, force_rebuild=args.force_rebuild)
    with open(args.cluster_file, "r") as f:
        for cid, line in enumerate(f):
            doc_ids = line.split(",")
            for did in doc_ids:
                index.add_item(cid, did, 1.0)
    index.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cluster_file", type=str, required=False)
    parser.add_argument("--index_dir", type=str, required=False)
    parser.add_argument("--index_name", type=str, required=False, default="invert_index")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--save_method", type=str, default="pickle", required=False)
    args = parser.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    
    build_invert_index(args)
    # ii = InvertIndex(args.index_dir, args.index_name, save_method=args.save_method)
    # print(ii.index_ids)