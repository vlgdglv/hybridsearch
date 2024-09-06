import json
import struct
import numpy as np
from tqdm import tqdm

def convert_query_ids_to_bin(args):
    with open(args.corpus_path, "r") as fr, open(args.output_path, "wb") as fw:
        lines = fr.readlines()
        total_query = len(lines)
        fw.write(struct.pack("I", total_query))
        for line in tqdm(fr):
            content = json.loads(line.strip())
            query_ids = np.array(content["text"], dtype=np.int32)
            query_value = np.array(content["value"] if "value" in content else [1.0 for _ in range(len(query_ids))], dtype=np.float32)
            
            fw.write(struct.pack("I", len(query_ids)))
            ids_size, values_size = query_ids.nbytes, query_value.nbytes
            fw.write(struct.pack("I", ids_size))
            fw.write(query_ids.tobytes())
            fw.write(struct.pack("I", values_size))
            fw.write(query_value.tobytes())


if __name__ == "__main__":
    pass