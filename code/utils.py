import torch
import numpy as np
import random
import pandas as pd

# from transformers import


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Summary
def load_data(data_path: int):
    pass


def tokenized_dataset(dataset: pd.DataFrame, tokenizer):
    pass


class SUM_Dataset:
    pass


def compute_metrics():
    pass
