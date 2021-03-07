import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model.binary_lstm import BinaryLSTMClassifier
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
import pickle as pkl
from utils.tokenizer import GloveTokenizer
from copy import deepcopy
import argparse
from utils.scheduler import get_cosine_schedule_with_warmup
import utils.nn_utils as nn_utils
from utils.file_logger import get_file_logger
from utils.tokenizer import *

# Argument parser for emo classifier
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

if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

logger = get_file_logger(args.log_path)  # Note: this is ugly, but I am lazy

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

# Seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

glove_tokenizer = GloveTokenizer(PAD_LEN)
glove_tokenizer.load_dict("/local/ssd_1/chengzhang/SA_dialog/dialogue/datas/nlpcc_dict.pkl")

emo_classifier = BinaryLSTMClassifier(
    emb_dim=SRC_EMB_DIM,
    vocab_size=34177,
    num_label=6,
    hidden_dim=SRC_HIDDEN_DIM,
    attention_mode=ATTENTION,
    args=args
    )
print('loading emo classifier...')
emo_classifier_list = []
for i in range(1,6):
    emo_classifier = BinaryLSTMClassifier(
        emb_dim=SRC_EMB_DIM,
        vocab_size=34177,
        num_label=6,
        hidden_dim=SRC_HIDDEN_DIM,
        attention_mode=ATTENTION,
        args=args
        )
    emo_classifier.load_state_dict(torch.load('saved_model/emotion_classifier' + str(i) + '.pt'))
    emo_classifier.cuda()
    emo_classifier_list.append(emo_classifier)
    
print('Done')

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
               
def inference_emotion(source, selected_emotion, batch_size):
    data_set = TestDataReader(source, 20)
    data_loader = DataLoader(data_set, batch_size=batch_size*3, shuffle=False)
    all_preds = []
    for i in range(len(emo_classifier_list)):
        emo_classifier = emo_classifier_list[i]
        emo_classifier.cuda()
        emo_classifier.eval()
        preds = []
        for i, (src, src_len) in enumerate(data_loader):
            with torch.no_grad():
                decoder_logit = emo_classifier(src.cuda(), src_len.cuda())
                softmax = nn.Softmax(dim=1)
                prob = softmax(decoder_logit)
                for i in range(batch_size):
                    preds.append(prob[i][int(selected_emotion)])
                del decoder_logit
        all_preds.append(preds)
    all_preds = np.asarray(all_preds)
    result = all_preds.mean(axis=0)

    final_result = []
    for num in result:
        final_result.append([num])
    return np.asarray(result)
