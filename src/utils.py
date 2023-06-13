'''
In this file a few helper functions are defined.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)

def calc_f1_score(pred, ans, tag2idx):
    """
    Calculates the F1 score of the predicted sequence and answer sequence.

    Args:
        pred (list): The predicted sequence.
        ans (list): The answer sequence.

    Returns:
        float: The F1 score.
    """
    # Calculate the true positives, false positives, and false negatives
    tp = fp = fn = 0
    # flatten the input to process batch dimension
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if ans[i][j] != tag2idx['<PAD>']:
                if pred[i][j] == ans[i][j] and pred[i][j] != tag2idx['O']:
                    tp += 1
                elif pred[i][j] != ans[i][j] and pred[i][j] != tag2idx['O']:
                    fp += 1
                elif pred[i][j] != ans[i][j] and ans[i][j] != tag2idx['O']:
                    fn += 1

    # Calculate the precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1