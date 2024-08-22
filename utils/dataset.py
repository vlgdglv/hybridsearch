import os
import pickle

class NQDataset:
    def __init__(self, data_path):
        self.data_path = data_path

        suffix = data_path.split(".")[-1]
        if suffix == "tsv":
            ids, text = [], []
            with open(data_path, "r") as f:
                for line in f:
                    line = line.strip().split("\t")
                    ids.append(line[0])
                    text.append(line[1])
            self.ids = ids
            self.text = text
        elif suffix in ["pt", "pkl"]:
            ids, text = pickle.load(open(data_path, "rb"))
        else:
            raise NotImplementedError
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.text[idx]
