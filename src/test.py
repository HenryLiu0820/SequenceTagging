from utils import *
import torch

# random int tensor
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag2idx = {"B": 0, "I": 1, "O": 2, "<PAD>": 5}
a = torch.randint(0, 10, (3, 4))
b = torch.randint(0, 10, (3, 4))
f1 = calc_f1_score(a, b, tag2idx)
print(f1)