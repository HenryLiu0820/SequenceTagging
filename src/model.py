'''
This python file contains the implementation of a BiLSTM-CRF model for sequence tagging.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, tag2idx, start_tag, stop_tag):
        super(BiLSTM_CRF, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.start_tag = start_tag
        self.stop_tag = stop_tag

        # tag related, tag2idx is a lookup table for different tags
        self.tag2idx = tag2idx
        self.tagset_size = len(tag2idx)

        # layers and modules
        self.word_embeds = nn.Embedding(vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # maps the LSTM output to the tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # transition matrix
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # make sure that the start and end transitions are never trained
        self.transitions.data[tag2idx[self.start_tag], :] = -10000
        self.transitions.data[:, tag2idx[self.stop_tag]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        '''
        Initialize the hidden state of the LSTM
        '''
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    

    def get_features(self, sentence):
        '''
        Get the feature output of a given sentence

        Input:
            sentence: a list of word indices
        Output:
            features: a tensor of shape (seq_len, tagset_size)
        '''
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.LSTM(embeds, self.hidden)

        # feature projection
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        features = self.hidden2tag(lstm_out)

        return features
    
    def calc_scores(self, feats):
        '''
        Calculate the scores for all possible paths to perform forward pass.
        Input:
            feats: the emission score marix, a tensor of shape (seq_len, tagset_size)
        Output:
            alpha: log sum exp score of all possible paths log(exp(S1)+log(exp(S2)+...+log(exp(Sn))
        '''
        # initialize the forward variables
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score
        init_alphas[0][self.tag2idx[self.start_tag]] = 0.

        # wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # iterate through the sentence
        for feat in feats:
            alphas_t = []  # the forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # calculate the emission scores, shape (1, tagset_size)
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # calculate the transition scores, shape (1, tagset_size), 
                # the ith entry is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # calculate the total score. 
                # the ith entry stores the score of the best tag sequence so far that ends with next_tag
                next_tag_var = forward_var + trans_score + emit_score
                # calculate the log sum exp for all the scores
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag2idx[self.stop_tag]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def decode(self, feats):
        '''
        Decode the best tag sequence for a given sentence, using Viterbi algorithm.
        Input:
            feats: the emission score marix, a tensor of shape (seq_len, tagset_size)
        '''
        backpointers = []

        # initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag2idx[self.start_tag]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step, plus the score of transitioning
                # from tag i to next_tag
                # we don't include the emission scores here because the max does not depend on them
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[self.stop_tag]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # pop off the start tag
        start = best_path.pop()
        assert start == self.tag2idx[self.start_tag]
        best_path.reverse()

        return path_score, best_path
    
    def score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.start_tag]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2idx[self.stop_tag], tags[-1]]
        return score
    
    def neg_log_likelihood(self, sentence, tags):
        '''
        Calculate the negative log likelihood
        Input:
            sentence: a list of word indices
            tags: a list of tags
        '''
        feats = self.get_features(sentence)
        forward_score = self.calc_scores(feats)         # all path score
        gold_score = self.score_sentence(feats, tags)   # score of actual path
        return forward_score - gold_score
    
    def forward(self, sentence):
        '''
        Forward pass
        Input:
            sentence: a list of word indices
        '''
        # get the emission scores from the BiLSTM
        lstm_feats = self.get_features(sentence)

        # find the best path, given the features
        score, tag_seq = self.decode(lstm_feats)
        return score, tag_seq