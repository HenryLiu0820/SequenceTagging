from utils import *
from parse_args import *
import os
import sys
import numpy as np
import pickle
import random
from tqdm import tqdm
import torch
from torch.optim import SGD
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.utils.data import DataLoader
from dataloader import *
from model import *

if __name__ == '__main__':

    ############################## 1. Parse arguments ################################
    print('parsing arguments...')
    args = parse_args()

    # create the checkpoint directory if it does not exist
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)


    if args.print_tofile == 'True':
        # Open files for stdout and stderr redirection
        stdout_file = open(os.path.join(args.ckpt_path, 'stdout.log'), 'w')
        stderr_file = open(os.path.join(args.ckpt_path, 'stderr.log'), 'w')
        # Redirect stdout and stderr to the files
        sys.stdout = stdout_file
        sys.stderr = stderr_file

    save_args_to_file(args, os.path.join(args.ckpt_path, 'args.json'))

    # print args
    print(args)

    ############################## 2. preprocess and loading the training data ################################
    if args.load == 'False':
        print('preprocessing the data...')
        train_text, train_tag, dev_text, dev_tag, test_text = load_data(args.datadir)
        word2idx, tag2idx, train_text_seq, train_tag_seq, dev_text_seq, dev_tag_seq, test_text_seq = preprocess(train_text, train_tag, dev_text, dev_tag, test_text, args)
        print('preprocessing finished')

    # load the training data
    print('loading the data...')
    save_path = os.path.join(args.datadir, 'processed_data')
    word2idx = pickle.load(open(os.path.join(save_path, 'word2idx.dat'), 'rb'))
    tag2idx = pickle.load(open(os.path.join(save_path, 'tag2idx.dat'), 'rb'))
    train_text_seq = pickle.load(open(os.path.join(save_path, 'train_text_seq.dat'), 'rb'))
    train_tag_seq = pickle.load(open(os.path.join(save_path, 'train_tag_seq.dat'), 'rb'))
    dev_text_seq = pickle.load(open(os.path.join(save_path, 'dev_text_seq.dat'), 'rb'))
    dev_tag_seq = pickle.load(open(os.path.join(save_path, 'dev_tag_seq.dat'), 'rb'))
    test_text_seq = pickle.load(open(os.path.join(save_path, 'test_text_seq.dat'), 'rb'))
    seq_len = len(train_text_seq[0])
    print('loading finished, vocabulary size: {}, sequence length: {}'.format(args.vocab_size, seq_len))


    ############################## 3. build the model ################################
    start_tag = '<START>'
    stop_tag = '<STOP>'
    # add two tags to the tag2idx dictionary
    tag2idx[start_tag] = len(tag2idx)
    tag2idx[stop_tag] = len(tag2idx)

    modelpath = os.path.join(args.ckpt_path, '{}.pt'.format(args.name))
    model = BiLSTM_CRF(args.vocab_size, args.embed_dim, args.hidden_dim, tag2idx, start_tag, stop_tag)
    model.train()
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if args.cuda == 'True':
        model.cuda()

    # use SGD as the optimizer
    optim = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    training_stats = {
        'epoch': [],
        'train_loss': [],
    }

    # flush the output
    sys.stdout.flush()

    ############################## 4. train the model ################################
    for epoch in range(1, args.epoch + 1):
        dataset = TaggingDataset(train_text_seq, train_tag_seq)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print('dataset size: {}'.format(len(dataset)))
        print('batch num: {}'.format(len(dataloader)))
        train_loss = 0
        loss_o = 0
        loss_n = 0
        step = 0
        avg_err = 0
        print('Starting epoch: {}'.format(epoch))
        for _, (seqs, tags) in enumerate(dataloader):
            step += 1
            if epoch == 1 and step <= 1:
                print('seqs: {}, shape: {}'.format(seqs, seqs.shape))
                print('tags: {}, shape: {}'.format(tags, tags.shape))
            sys.stdout.flush()
            model.zero_grad()
            if args.cuda == 'True':
                seqs, tags = seqs.cuda(), tags.cuda()
            loss = model.neg_log_likelihood(seqs, tags)
            train_loss += loss.item()
            loss.backward()
            optim.step()

            # print the training stats
            if step % 1000 == 0:
                in_embed = model.get_embeddings('in')

        train_loss /= step
        loss_o /= step
        loss_n /= step
        print('Finished Epoch: {}, train_loss: {}, loss_o: {}, loss_n: {}, avg error: {}'.format(epoch, train_loss, loss_o, loss_n, avg_err))

        # update training stats
        training_stats['epoch'].append(epoch)
        training_stats['train_loss'].append(train_loss)
        training_stats['loss_o'].append(loss_o)
        training_stats['loss_n'].append(loss_n)
        training_stats['avg_err'] = avg_err

        # flush the output
        sys.stdout.flush()


    np.save(os.path.join(args.ckpt_path, "training_stats.npy"), training_stats)
    
    if args.print_tofile == 'True':
        # Close the files to flush the output
        stdout_file.close()
        stderr_file.close()