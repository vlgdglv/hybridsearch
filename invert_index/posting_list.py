import os

# 
class PostingList:
    def __init__(self):
        self.current_idx = 0

        self.list_item = []
        self.list_len = 0

    def append(self, item):
        self.list_item.append(item)
        self.list_len += 1

    def set_posting(self, posting_list):
        self.list_item = posting_list
        self.list_len = len(posting_list)

    def __len__(self):
        return self.list_len

    def __getitem__(self, idx):
        return self.list_item[idx]