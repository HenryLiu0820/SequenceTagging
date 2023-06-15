import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for sequence tagging')

    # basic arguments
    parser.add_argument('--name', type=str, default='BiLSTM-CRF', help='name of the experiment')
    parser.add_argument('--load', type=str, default='False', help='if load training data from local file')
    parser.add_argument('--print_tofile', type=str, default='True', help='if print log content to file')
    parser.add_argument('--ckpt_path', type=str, default='./data', help='directory to store and load the checkpoints')
    parser.add_argument('--datadir', type=str, default='/Users/henryliu/Desktop/Henry/学习/untitled folder/大三/大三下/自然语言处理/labs/project3', help='directory to training/testing data')
    parser.add_argument('--wandb-on', type=str, default='False', help='if use wandb for logging')

    # preprocessing related
    parser.add_argument('--seq_len', type=int, default=8, help='tagging sequence length')
    parser.add_argument('--vocab_size', type=int, default=50000, help='max vocabulary size')
    # training related
    parser.add_argument('--embed_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=300, help="hidden dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--cuda', type=str, default='False', help="use CUDA")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay")

    args = parser.parse_args()
    return args