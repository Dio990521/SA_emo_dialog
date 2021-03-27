import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl
import copy
from tqdm import tqdm
import argparse
from copy import deepcopy
from model.binary_bert import BinaryBertClassifier
from transformers import BertTokenizer, AdamW
import random
from utils import nn_utils

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=16, type=int, help="batch size")
parser.add_argument('--postname', default='', type=str, help="post name")
parser.add_argument('--gamma', default=0.2, type=float, help="post name")
parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
parser.add_argument('--dataset', default='sem18', type=str, choices=['sem18', 'goemotions', 'bmet'])
parser.add_argument('--criterion', default='jaccard', type=str,
                    help='criterion to prevent overfitting, currently support f1 and loss')
parser.add_argument('--bert', default='base', type=str, help="bert size [base/large]")
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--warmup_epoch', default=2, type=int, help='')
parser.add_argument('--stop_epoch', default=10, type=int, help='')
parser.add_argument('--max_epoch', default=20, type=int, help='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--en_lr', type=float, default=5e-5)
parser.add_argument('--de_lr', default=5e-5, type=float, help="decoder learning rate")
parser.add_argument('--attention', default='dot', type=str, help='general/mlp/dot')
parser.add_argument('--de_dim', default=400, type=int, help="dimension")
parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--attention_dropout', default=0.25, type=float, help='dropout rate')
parser.add_argument('--input_feeding', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--huang_init', action='store_true')
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--unify_decoder', action='store_true')
parser.add_argument('--eval_every', type=int, default=500)
parser.add_argument('--patience', default=3, type=int, help='dropout rate')
parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
parser.add_argument('--en_de_activate_function', default='tanh', type=str)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)


args = parser.parse_args(args=[])


if args.bert == 'base':
    BERT_MODEL = 'bert-base-chinese'
    SRC_HIDDEN_DIM = 768
else:
    raise ValueError('Specified BERT model NOT supported!!')

NUM_FOLD = 5
ENCODER_LEARNING_RATE = args.en_lr
PAD_LEN = 50
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
TGT_HIDDEN_DIM = args.de_dim
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = args.stop_epoch

# Get BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
# BERT optimizer setup
max_grad_norm = 1.0
# Seed
RANDOM_SEED = args.seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print('loading emo classifier...')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
emo_classifier = torch.load("/local/ssd_1/stc/nlpcc_bert.pt")
print('Done')

class TestDataReader(Dataset):
    def __init__(self, X, pad_len, max_size=None):
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.tokens = []
        self.token_masks = []
        self.pad_int = 0
        self.__read_data(X)

    def __read_data(self, data_list):
        for X in data_list:

            X = tokenizer.tokenize(X)
            X = ['[CLS]'] + X + ['[SEP]']
            X = tokenizer.convert_tokens_to_ids(X)
            X_len = len(X)

            if len(X) > self.pad_len:
                X = X[:self.pad_len]
                mask = [1] * self.pad_len
            else:
                X = X + [self.pad_int] * (self.pad_len - len(X))
                mask = [1] * X_len + [0] * (self.pad_len - X_len)

            self.tokens.append(X)
            self.token_masks.append(mask)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx]), \
               torch.LongTensor(self.token_masks[idx])


class TrainDataReader(TestDataReader):
    def __init__(self, X, y, pad_len, max_size=None):
        super(TrainDataReader, self).__init__(X, pad_len, max_size)
        self.y = []
        self.__read_target(y)

    def __read_target(self, y):
        self.y = y

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx]), \
               torch.LongTensor(self.token_masks[idx]), \
               torch.LongTensor([self.y[idx]])


def inference_emotion(source, selected_emotion, batch_size):
    emo_classifier.cuda()
    emo_classifier.eval()
    data_set = TestDataReader(source, 20)
    data_loader = DataLoader(data_set, batch_size=batch_size*3, shuffle=False)
    pred_list = []
    for _, (_data, _mask) in enumerate(data_loader):
        with torch.no_grad():
            decoder_logit = emo_classifier(_data.cuda(), _mask.cuda())
            softmax = nn.Softmax(dim=1)
            prob = softmax(decoder_logit)
            for i in range(batch_size):
                    pred_list.append(prob[i][int(selected_emotion)])
            del decoder_logit

    return np.asarray(pred_list)