'''
This file contains the loading, preprocessing data and dataloader preperation.
'''
import numpy as np
import pandas as pd
import os
import pickle
import collections
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(datadir):
    '''
    Load the train & test data from local txt files
    '''

    # directories
    train_text_file = os.path.join(datadir, 'train.txt')
    train_tag_file = os.path.join(datadir, 'train_TAG.txt')
    dev_text_file = os.path.join(datadir, 'dev.txt')
    dev_tag_file = os.path.join(datadir, 'dev_TAG.txt')
    test_file = os.path.join(datadir, 'test.txt')

    # load data
    with open(train_text_file, 'r') as f:
        train_text = f.read()
    f.close()
    with open(train_tag_file, 'r') as f:
        train_tag = f.read()
    f.close()
    with open(dev_text_file, 'r') as f:
        dev_text = f.read()
    f.close()
    with open(dev_tag_file, 'r') as f:
        dev_tag = f.read()
    f.close()
    with open(test_file, 'r') as f:
        test_text = f.read()
    f.close()

    return train_text, train_tag, dev_text, dev_tag, test_text

def fix_seq_len(text, seq_len, pad_token, dict, is_text=True):
    '''
    This function process a list of strings, 
    seperates the strings into tokens and pad each string to the same length
    '''
    lines = text.split('\n')
    result = []
    for line in lines:
        if is_text:
            line = line.strip().lower()
        else:
            line = line.strip()

        line = line.split(' ')
        line_len = len(line)
        if line_len > seq_len:
            line = line[: seq_len]
            line_len = seq_len
        assert line_len <= seq_len
        line = line + [pad_token] * (seq_len - line_len)
        result.append([dict.get(word, 0) for word in line])

    return result

def preprocess(train_text, train_tag, dev_text, dev_tag, test_text, args):
    '''
    Process the train set and dev test, and align sequences to fixed length
    args used: vocab_size, datadir
    '''
    vocab_size = args.vocab_size
    # process train text, construct vocabulary
    train_corpus = train_text.replace('\n', ' ')
    train_corpus = train_corpus.strip().lower()
    train_corpus = train_corpus.split(' ')
    print('Finished extracting words. Total words: {}'.format(len(train_corpus)))

    # count the frequency of each word, and preserve only the most frequent words
    counter = collections.Counter(train_corpus)
    most_freq = counter.most_common(vocab_size)
    vocabulary = [word for word, _ in most_freq]
    vocab_size = min(vocab_size, len(vocabulary))

    # add an <UNK> to the front and a <PAD> to the end
    vocabulary = ['<UNK>'] + vocabulary[: len(vocabulary) - 2] + ['<PAD>']
    assert len(vocabulary) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocabulary)}

    # process train tags
    train_tags = list(set(train_tag.replace('\n', ' ').split(' ')))
    train_tags = ['<UNK>'] + train_tags + ['<PAD>']
    tag2idx = {tag: idx for idx, tag in enumerate(train_tags)}

    # tokenize the train text and train tag
    train_text_lines = train_text.split('\n')
    train_tag_lines = train_tag.split('\n')

    # set sequence len 
    seq_len = args.seq_len

    train_text_seq = fix_seq_len(train_text, seq_len, '<PAD>', word2idx, is_text=True)
    train_tag_seq = fix_seq_len(train_tag, seq_len, '<PAD>', tag2idx, is_text=False)

    print('Sequence length of train text: {}'.format(seq_len))

    # process dev text
    dev_text_seq = fix_seq_len(dev_text, seq_len, '<PAD>', word2idx, is_text=True)
    dev_tag_seq = fix_seq_len(dev_tag, seq_len, '<PAD>', tag2idx, is_text=False)

    # process test text
    test_text_seq = fix_seq_len(test_text, seq_len, '<PAD>', word2idx, is_text=True)

    # save the processed data
    save_path = os.path.join(args.datadir, 'processed_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pickle.dump(word2idx, open(os.path.join(save_path, 'word2idx.dat'), 'wb'))
    pickle.dump(tag2idx, open(os.path.join(save_path, 'tag2idx.dat'), 'wb'))
    pickle.dump(train_text_seq, open(os.path.join(save_path, 'train_text_seq.dat'), 'wb'))
    pickle.dump(train_tag_seq, open(os.path.join(save_path, 'train_tag_seq.dat'), 'wb'))
    pickle.dump(dev_text_seq, open(os.path.join(save_path, 'dev_text_seq.dat'), 'wb'))
    pickle.dump(dev_tag_seq, open(os.path.join(save_path, 'dev_tag_seq.dat'), 'wb'))
    pickle.dump(test_text_seq, open(os.path.join(save_path, 'test_text_seq.dat'), 'wb'))
    print('tag2idx', tag2idx)
    print('word2idx len: {}'.format(len(word2idx)))
    print('Finished saving processed data')

    return word2idx, tag2idx, train_text_seq, train_tag_seq, dev_text_seq, dev_tag_seq, test_text_seq

class TaggingDataset(Dataset):
    def __init__(self, text_seq, tag_seq):
        super().__init__()
        self.text_seq = torch.tensor(text_seq, dtype=torch.int32)
        self.tag_seq = torch.tensor(tag_seq, dtype=torch.int32)

    def __len__(self):
        return len(self.text_seq)
    
    def __getitem__(self, idx):
        return self.text_seq[idx], self.tag_seq[idx]
