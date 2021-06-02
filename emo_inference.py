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

from model.binary_lstm import BinaryLSTMClassifier
from utils.tokenizer import GloveTokenizer

#-------bert--------------

#parser = argparse.ArgumentParser(description='Options')
#parser.add_argument('--batch_size', default=16, type=int, help="batch size")
#parser.add_argument('--postname', default='', type=str, help="post name")
#parser.add_argument('--gamma', default=0.2, type=float, help="post name")
#parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
#parser.add_argument('--dataset', default='sem18', type=str, choices=['sem18', 'goemotions', 'bmet'])
#parser.add_argument('--criterion', default='jaccard', type=str,
#                    help='criterion to prevent overfitting, currently support f1 and loss')
#parser.add_argument('--bert', default='base', type=str, help="bert size [base/large]")
#parser.add_argument('--scheduler', action='store_true')
#parser.add_argument('--warmup_epoch', default=2, type=int, help='')
#parser.add_argument('--stop_epoch', default=10, type=int, help='')
#parser.add_argument('--max_epoch', default=20, type=int, help='')
#parser.add_argument('--seed', type=int, default=0)
#parser.add_argument('--en_lr', type=float, default=5e-5)
#parser.add_argument('--de_lr', default=5e-5, type=float, help="decoder learning rate")
#parser.add_argument('--attention', default='dot', type=str, help='general/mlp/dot')
#parser.add_argument('--de_dim', default=400, type=int, help="dimension")
#parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
#parser.add_argument('--attention_dropout', default=0.25, type=float, help='dropout rate')
#parser.add_argument('--input_feeding', action='store_true')
#parser.add_argument('--glorot_init', action='store_true')
#parser.add_argument('--huang_init', action='store_true')
#parser.add_argument('--normal_init', action='store_true')
#parser.add_argument('--unify_decoder', action='store_true')
#parser.add_argument('--eval_every', type=int, default=500)
#parser.add_argument('--patience', default=3, type=int, help='dropout rate')
#parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
#parser.add_argument('--en_de_activate_function', default='tanh', type=str)
#parser.add_argument('--log_path', type=str, default=None)
#parser.add_argument('--output_path', type=str, default=None)
#
#
#args = parser.parse_args(args=[])
#
#
#if args.bert == 'base':
#    BERT_MODEL = 'bert-base-chinese'
#    SRC_HIDDEN_DIM = 768
#else:
#    raise ValueError('Specified BERT model NOT supported!!')

#NUM_FOLD = 5
#ENCODER_LEARNING_RATE = args.en_lr
#PAD_LEN = 50
#MIN_LEN_DATA = 3
#BATCH_SIZE = args.batch_size
#TGT_HIDDEN_DIM = args.de_dim
#ATTENTION = args.attention
#PATIENCE = args.patience
#WARMUP_EPOCH = args.warmup_epoch
#STOP_EPOCH = args.stop_epoch

## Get BERT tokenizer
#tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
## BERT optimizer setup
#max_grad_norm = 1.0
## Seed
#RANDOM_SEED = args.seed
#torch.manual_seed(RANDOM_SEED)
#torch.cuda.manual_seed(RANDOM_SEED)
#torch.cuda.manual_seed_all(RANDOM_SEED)
#np.random.seed(RANDOM_SEED)
#random.seed(RANDOM_SEED)
#
#
#print('loading bert classifier...')
#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#emo_classifier = torch.load("/local/ssd_1/stc/nlpcc_bert.pt")
#print('Done')

#class TestDataReader(Dataset):
#    def __init__(self, X, pad_len, max_size=None):
#        self.glove_ids_len = []
#        self.pad_len = pad_len
#        self.tokens = []
#        self.token_masks = []
#        self.pad_int = 0
#        self.__read_data(X)
#
#    def __read_data(self, data_list):
#        for X in data_list:
#
#            X = tokenizer.tokenize(X)
#            X = ['[CLS]'] + X + ['[SEP]']
#            X = tokenizer.convert_tokens_to_ids(X)
#            X_len = len(X)
#
#            if len(X) > self.pad_len:
#                X = X[:self.pad_len]
#                mask = [1] * self.pad_len
#            else:
#                X = X + [self.pad_int] * (self.pad_len - len(X))
#                mask = [1] * X_len + [0] * (self.pad_len - X_len)
#
#            self.tokens.append(X)
#            self.token_masks.append(mask)
#
#    def __len__(self):
#        return len(self.tokens)
#
#    def __getitem__(self, idx):
#        return torch.LongTensor(self.tokens[idx]), \
#               torch.LongTensor(self.token_masks[idx])
#
#
#class TrainDataReader(TestDataReader):
#    def __init__(self, X, y, pad_len, max_size=None):
#        super(TrainDataReader, self).__init__(X, pad_len, max_size)
#        self.y = []
#        self.__read_target(y)
#
#    def __read_target(self, y):
#        self.y = y
#
#    def __getitem__(self, idx):
#        return torch.LongTensor(self.tokens[idx]), \
#               torch.LongTensor(self.token_masks[idx]), \
#               torch.LongTensor([self.y[idx]])

# ------------------lstm-------------------------
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=32, type=int, help="batch size")
parser.add_argument('--pad_len', default=50, type=int, help="batch size")
parser.add_argument('--postname', default='', type=str, help="post name")
parser.add_argument('--gamma', default=0.2, type=float, help="post name")
parser.add_argument('--folds', default=5, type=int, help="num of folds")
parser.add_argument('--en_lr', default=5e-4, type=float, help="encoder learning rate")
parser.add_argument('--de_lr', default=5e-4, type=float, help="decoder learning rate")
parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
parser.add_argument('--dataset', default='nlpcc', type=str, choices=['sem18', 'goemotions', 'bmet', 'nlpcc'])
parser.add_argument('--en_dim', default=800, type=int, help="dimension")
parser.add_argument('--de_dim', default=400, type=int, help="dimension")
parser.add_argument('--criterion', default='micro', type=str, choices=['jaccard', 'macro', 'micro', 'h_loss'])
parser.add_argument('--glove_path', default='data/glove.840B.300d.txt', type=str)
parser.add_argument('--attention', default='bert', type=str, choices=['bert', 'attentive', 'None'])
parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
parser.add_argument('--encoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--attention_dropout', default=0.25, type=float, help='dropout rate')
parser.add_argument('--patience', default=10, type=int, help='dropout rate')
parser.add_argument('--download_elmo', action='store_true')
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--warmup_epoch', default=0, type=int, help='')
parser.add_argument('--stop_epoch', default=50, type=int, help='')
parser.add_argument('--max_epoch', default=100, type=int, help='')
parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--input_feeding', action='store_true')
parser.add_argument('--dev_split_seed', type=int, default=0)
parser.add_argument('--huang_init', action='store_true')
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--unify_decoder', action='store_true')
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--no_cross', action='store_true')
args = parser.parse_args(args=[])
SRC_EMB_DIM = 300
MAX_LEN_DATA = args.pad_len
PAD_LEN = MAX_LEN_DATA
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
CLIPS = 0.666
GAMMA = 0.5
SRC_HIDDEN_DIM = args.en_dim
TGT_HIDDEN_DIM = args.de_dim
VOCAB_SIZE = 60000
ENCODER_LEARNING_RATE = args.en_lr
DECODER_LEARNING_RATE = args.de_lr
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = args.stop_epoch
MAX_EPOCH = args.max_epoch
RANDOM_SEED = args.seed
print('loading lstm classifier...')
glove_tokenizer = GloveTokenizer(PAD_LEN)

EMOS = ['anger', 'disgust', 'happiness', 'like', 'sadness', 'none']
NUM_EMO = len(EMOS)

# Seed
RANDOM_SEED = args.seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def get_emotion(file, EMOS, EMOS_DIC):
    text_list = []
    label_list = []
    with open(file, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            text, emotion = line.split('\t')
            text_list.append(text)
            label_list.append(int(EMOS_DIC[emotion.rstrip("\n")]))
    return text_list, label_list


def load_NLPCC_data():
    EMOS_DIC = {}
    for idx, emo in enumerate(EMOS):
        EMOS_DIC[emo] = idx
    file1 = "/local/ssd_1/chengzhang/BiLSTM_emo_classifier/data/nlpcc/nlpcc2013_2014_adjust.txt"

    X, y = get_emotion(file1, EMOS, EMOS_DIC)
    X_train = X[:18000]
    y_train = y[:18000]
    X_dev = X[18000:19000]
    y_dev = y[18000:19000]
    X_test = X[19000:]
    y_test = y[19000:]
    X_train_dev = X_train + X_dev
    y_train_dev = y_train + y_dev
    # preprocess

    return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, 'nlpcc'
    
class TestDataReader(Dataset):
    def __init__(self, X, pad_len, max_size=None):
        self.glove_ids = []
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.build_glove_ids(X)

    def build_glove_ids(self, X):
        for src in X:
            glove_id, glove_id_len = glove_tokenizer.encode_ids_pad(src)
            self.glove_ids.append(glove_id)
            self.glove_ids_len.append(glove_id_len)

    def __len__(self):
        return len(self.glove_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]])


class TrainDataReader(TestDataReader):
    def __init__(self, X, y, pad_len, max_size=None):
        super(TrainDataReader, self).__init__(X, pad_len, max_size)
        self.y = []
        self.read_target(y)

    def read_target(self, y):
        self.y = y

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]]), \
               torch.LongTensor([self.y[idx]])

X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
    load_NLPCC_data()
    
ALL_TRAINING = X_train_dev + X_test
glove_tokenizer.build_tokenizer(ALL_TRAINING, vocab_size=VOCAB_SIZE)
print('Done')
#---------------------------------
def inference_emotion2(source, selected_emotion, batch_size):
    emo_classifier.cuda()
    emo_classifier.eval()
    data_set = TestDataReader(source, 20)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
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
    
    
def inference_emotion(source, selected_emotion, batch_size):
    model = BinaryLSTMClassifier(
        emb_dim=SRC_EMB_DIM,
        vocab_size=34177,
        num_label=NUM_EMO,
        hidden_dim=SRC_HIDDEN_DIM,
        attention_mode=ATTENTION,
        args=args
    )
    model.cuda()
    
    data_set = TestDataReader(source, 20)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    pred_list = []
    model.load_state_dict(torch.load("/local/ssd_1/chengzhang/SA_dialog/dialogue/saved_classifier/emotion_classifier1.pt"))
    model.eval()
    for _, (src, src_len) in enumerate(data_loader):
        with torch.no_grad():
        
            decoder_logit = model(src.cuda(), src_len.cuda())
            softmax = nn.Softmax(dim=1)
            prob = softmax(decoder_logit)
            for i in range(batch_size):
                pred_list.append(prob[i][int(selected_emotion)])
            del decoder_logit
        
    return np.asarray(pred_list)