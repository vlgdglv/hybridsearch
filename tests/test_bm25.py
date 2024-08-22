import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bm25.tokenizer import *

if __name__ == "__main__": 
    token = Tokenizer()
    token.train(["hello world", "hello world"], save_dir=".")

    q = "hello world, you shit"

    print(token.encode(q))