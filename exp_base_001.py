import time
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import copy

import re
from gensim.models import KeyedVectors
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler


data_dir = 'your data directory'
embed_dir = 'your embedding directory'

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name',
                    default='', type=str, required=True)

parser.add_argument('--exp_num',
                    default=999, type=int, required=True,
                    help='0-999: Experiment number. The log filename is exp{exp_num}.csv')

parser.add_argument('--seed',
                    default=123456, type=int, required=False,
                    help='Random seed for reproduciblity')

parser.add_argument('--split_seed_s',
                    default='100 1000', type=str, required=False,
                    help='String of space delimited integer(s): Random seeds for CV spliting')
parser.add_argument('--n_folds',
                    default=10, type=int, required=False,
                    help='Number of CV folds.')


parser.add_argument('--n_models',
                    default=6, type=int, required=False)
parser.add_argument('--epochs',
                    default=10, type=int, required=False)
parser.add_argument('--batch_size',
                    default=512, type=int, required=False)
parser.add_argument('--hidden_dim',
                    default=128, type=int, required=False)
parser.add_argument('--drop_last',
                    default=0, type=int, required=False,
                    help='Drop the last incomplete batch or not.')

# Preprocess
parser.add_argument('--lower',
                    default=0, type=int, required=False,
                    help='Lower question text.')
parser.add_argument('--clean_num',
                    default=2, type=int, required=False,
                    help='clean_numbers 0:None, 1:All, 2:Only for ggle word_vec.')
parser.add_argument('--maxlen',
                    default=50, type=int, required=False)
parser.add_argument('--max_features',
                    default=90000, type=int, required=False)
parser.add_argument('--trunc',
                    default='pre', type=str, required=False,
                    help="'pre' or 'post': remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences.")

# Embedding
parser.add_argument('--fix_embedding',
                    default=1, type=int, required=False)
parser.add_argument('--unk_uni',
                    default=1, type=int, required=False,
                    help='Initializer for unknown words.')
parser.add_argument('--n_embed',
                    default=2, type=int, required=False,
                    help='1, 2, 3, 4: Number of pretrained embeddings.')
parser.add_argument('--pme_relu',
                    default=1, type=int, required=False,
                    help='1 or 0: Use ReLU in PME or not.')

# GRU
parser.add_argument('--bidirectional',
                    default=1, type=int, required=False,
                    help='1 or 0: Bidirectional or not.')
parser.add_argument('--n_layers',
                    default=1, type=int, required=False,
                    help='Number of GRU layers.')

# EMA
parser.add_argument('--mu',
                    default=0.9, type=float, required=False,
                    help='EMA decay.')
parser.add_argument('--updates_per_epoch',
                    default=10, type=int, required=False,
                    help='Number of EMA updates per epoch.')

# DIR
parser.add_argument('--log_dir',
                    default='log/', type=str, required=False)

args = parser.parse_args()

with open(args.log_dir + f'exp{args.exp_num:03d}.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

t0 = time.time()


def seed_torch(seed=9999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tqdm.pandas()

exp_num = args.exp_num
log_fname = f'exp{exp_num:03d}.csv'

log_path = args.log_dir + log_fname


def get_param_size(model, trainable=True):
    if trainable:
        psize = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    else:
        psize = np.sum([np.prod(p.size()) for p in model.parameters()])
    return psize


def load_glove(word_index, max_features, unk_uni):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE = embed_dir + 'glove.840B.300d/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))

    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns glove', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


def load_wiki(word_index, max_features, unk_uni):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE = embed_dir + 'wiki-news-300d-1M/wiki-news-300d-1M.vec'
    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE) if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))

    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns wiki', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


def load_parag(word_index, max_features, unk_uni):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE = embed_dir + 'paragram_300_sl999/paragram_300_sl999.txt'
    embeddings_index = dict(get_coefs(*o.split(' '))
                            for o in open(EMBEDDING_FILE, encoding='utf8', errors='ignore')
                            if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))
    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns parag', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def load_ggle(word_index, max_features, unk_uni):
    EMBEDDING_FILE = embed_dir + 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    embed_size = embeddings_index.get_vector('known').size

    unknown_words = []
    nb_words = min(max_features, len(word_index))
    if unk_uni:
        embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
    else:
        embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        if word in embeddings_index:
            embedding_vector = embeddings_index.get_vector(word)
            embedding_matrix[i] = embedding_vector
        else:
            word_lower = word.lower()
            if word_lower in embeddings_index:
                embedding_matrix[i] = embeddings_index.get_vector(word_lower)
            else:
                unknown_words.append((word, i))

    print('\nTotal unknowns ggle', len(unknown_words))
    print(unknown_words[:10])

    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words


def load_all_embeddings(tokenizer, max_features, clean_num=False, unk_uni=True):
    word_index = tokenizer.word_index
    if clean_num == 2:
        ggle_word_index = {}
        for word, i in word_index.items():
            ggle_word_index[clean_numbers(word)] = i
    else:
        ggle_word_index = word_index

    embedding_matrix_1, u1 = load_glove(word_index, max_features, unk_uni)
    embedding_matrix_2, u2 = load_wiki(word_index, max_features, unk_uni)
    embedding_matrix_3, u3 = load_parag(word_index, max_features, unk_uni)
    embedding_matrix_4, u4 = load_ggle(ggle_word_index, max_features, unk_uni)
    embedding_matrix = np.concatenate((embedding_matrix_1,
                                       embedding_matrix_2,
                                       embedding_matrix_3,
                                       embedding_matrix_4), axis=1)
    del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4
    gc.collect()
    # with open('unknowns.pkl', 'wb') as f:
    #     pickle.dump({'glove': u1, 'wiki': u2, 'parag': u3, 'ggle': u4}, f)
    # print('Embedding:', embedding_matrix.shape)
    return embedding_matrix


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856
class EMA():
    def __init__(self, model, mu, level='batch', n=1):
        """
        level: 'batch' or 'epoch'
          'batch': Update params every n batches.
          'epoch': Update params every epoch.
        """
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.level = level
        self.n = n
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level is 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n

    def on_epoch_end(self, model):
        if self.level is 'epoch':
            self._update(model)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

    def forward(self, inputs):
        z, _ = torch.max(inputs, 1)
        return z

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GRUModel(nn.Module):
    """
    """
    def __init__(self, n_vocab, embed_dim, proj_dim, rnn_dim, n_layers, bidirectional, dense_dim,
                 padding_idx=0, pretrained_embedding=None, fix_embedding=True, pme_relu=True,
                 n_out=1):
        super(GRUModel, self).__init__()
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dense_dim = dense_dim
        self.n_out = n_out
        self.bidirectional = bidirectional
        self.fix_embedding = fix_embedding
        self.padding_idx = padding_idx
        if pretrained_embedding is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embedding, freeze=fix_embedding)
            self.embed.padding_idx = self.padding_idx
        else:
            self.embed = nn.Embedding(self.n_vocab, self.embed_dim, padding_idx=self.padding_idx)
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.proj_act = nn.ReLU() if pme_relu else None
        self.gru = nn.GRU(proj_dim, rnn_dim, self.n_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.pooling = GlobalMaxPooling1D()
        in_dim = 2 * rnn_dim if self.bidirectional else rnn_dim
        self.dense = nn.Linear(in_dim, dense_dim)
        self.dense_act = nn.ReLU()
        self.out_linear = nn.Linear(dense_dim, n_out)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.find('embed') > -1:
                continue
            elif name.find('weight') > -1 and len(param.size()) > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs):
        # inputs: (bs, max_len)
        x = self.embed(inputs)
        x = self.proj(x)
        if self.proj_act is not None:
            x = self.proj_act(x)
        x, hidden = self.gru(x)
        x = self.pooling(x)
        x = self.dense_act(self.dense(x))
        x = self.out_linear(x)
        return x

    def predict(self, dataloader):
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                X_batch, = batch
                preds.append(self.forward(X_batch).data.cpu())
        return torch.cat(preds)

    def predict_proba(self, dataloader):
        return torch.sigmoid(self.predict(dataloader)).data.numpy()


def get_dataloader(x, y=None, weights=None, num_samples=None, batch_size=32,
                   dtype_x=torch.float, dtype_y=torch.float, training=True,
                   drop_last=False):
    x_tensor = torch.tensor([x_1 for x_1 in x], dtype=dtype_x)
    if y is None:
        data = TensorDataset(x_tensor)
    else:
        y_tensor = None if y is None else torch.tensor([y_1 for y_1 in y], dtype=dtype_y)
        data = TensorDataset(x_tensor, y_tensor)
    if training:
        if weights is None:
            sampler = RandomSampler(data)
        else:
            sampler = WeightedRandomSampler(weights, num_samples)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, shuffle=False, batch_size=batch_size,
                            drop_last=drop_last)
    return dataloader


def f1_best(y, pred, thresh_s=None):
    if thresh_s is None:
        thresh_s = np.linspace(0.1, 0.5, 41)
    best_f1 = 0
    best_thresh = 0
    for thresh in thresh_s:
        f1 = f1_score(y, (pred > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_f1, best_thresh


def setup_emb(tr_X, max_features=50000, clean_num=2, unk_uni=True):
    tokenizer = Tokenizer(num_words=max_features, lower=False, filters='')
    tokenizer.fit_on_texts(tr_X)
    print('len(vocab)', len(tokenizer.word_index))
    embedding_matrix = load_all_embeddings(tokenizer, max_features=max_features,
                                           clean_num=clean_num, unk_uni=unk_uni)
    # np.save(embed_path, embedding_matrix)
    return tokenizer, embedding_matrix


puncts = ',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'


def clean_text(x, puncts=puncts):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def prepare_data(train_df, test_df, maxlen, max_features, trunc='pre',
                 lower=False, clean_num=2, unk_uni=True):
    train_df = train_df.copy()

    # lower
    if lower:
        train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: x.lower())
        test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: x.lower())

    # Clean the text
    train_df['question_text'] = train_df['question_text'].progress_apply(
        lambda x: clean_text(x))
    test_df['question_text'] = test_df['question_text'].progress_apply(
        lambda x: clean_text(x))

    # Clean numbers
    if clean_num == 1:
        train_df['question_text'] = train_df['question_text'].progress_apply(
            lambda x: clean_numbers(x))
        test_df['question_text'] = test_df['question_text'].progress_apply(
            lambda x: clean_numbers(x))

    # fill up the missing values
    train_df['question_text'] = train_df['question_text'].fillna('_##_')
    test_df['question_text'] = test_df['question_text'].fillna('_##_')

    train_X = train_df['question_text'].values
    test_X = test_df['question_text'].values

    tokenizer, embedding_matrix = setup_emb(train_X,
                                            max_features=max_features,
                                            clean_num=clean_num, unk_uni=unk_uni)

    tr_X_ids = tokenizer.texts_to_sequences(train_X)
    tr_X_padded = pad_sequences(tr_X_ids, maxlen=maxlen, truncating=trunc)
    test_X_ids = tokenizer.texts_to_sequences(test_X)
    test_X_padded = pad_sequences(test_X_ids, maxlen=maxlen, truncating=trunc)
    embedding_matrix = torch.Tensor(embedding_matrix)

    return tr_X_padded, test_X_padded, embedding_matrix, tokenizer


def run_epoch(model, dataloader, optimizer, callbacks=None,
              criterion=nn.BCEWithLogitsLoss(), verbose_step=10000):
    t1 = time.time()
    tr_loss = 0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        x_batch, y_batch = batch
        model.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs[:, 0], y_batch.float())
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        if callbacks is not None:
            for func in callbacks:
                func.on_batch_end(model)
        if (step + 1) % verbose_step == 0:
            loss_now = tr_loss / (step + 1)
            print(f'step:{step+1} loss:{loss_now:.7f} time:{time.time() - t1:.1f}s')
    if callbacks is not None:
        for func in callbacks:
            func.on_epoch_end(model)
    return tr_loss / (step + 1)


ids_s = [list(range(300)), list(range(300, 600)),
         list(range(600, 900)), list(range(900, 1200))]

cols_s_dict = {1: [ids_s[0], ids_s[1], ids_s[2], ids_s[3]],
               2: [ids_s[0] + ids_s[1],
                   ids_s[0] + ids_s[2],
                   ids_s[1] + ids_s[2],
                   ids_s[0] + ids_s[3],
                   ids_s[1] + ids_s[3],
                   ids_s[2] + ids_s[3]],
               3: [ids_s[0] + ids_s[1] + ids_s[2],
                   ids_s[0] + ids_s[1] + ids_s[3],
                   ids_s[0] + ids_s[2] + ids_s[3],
                   ids_s[1] + ids_s[2] + ids_s[3]],
               4: [ids_s[0] + ids_s[1] + ids_s[2] + ids_s[3]]}


n_folds = args.n_folds
split_seed_s = [int(val) for val in args.split_seed_s.split()]

clean_num = args.clean_num
lower = args.lower
maxlen = args.maxlen
max_features = args.max_features
trunc = args.trunc

n_vocab = max_features

drop_last = args.drop_last == 1
n_models = args.n_models
epochs = args.epochs
batch_size = args.batch_size

hidden_dim = args.hidden_dim

fix_embedding = args.fix_embedding == 1
unk_uni = args.unk_uni
n_embed = args.n_embed
embed_dim = n_embed * 300
proj_dim = hidden_dim
cols_s = cols_s_dict[n_embed]
pme_relu = args.pme_relu == 1

bidirectional = args.bidirectional == 1
n_layers = args.n_layers
rnn_dim = hidden_dim
dense_dim = 2 * rnn_dim if bidirectional else rnn_dim

thresh_s = np.linspace(0.2, 0.5, 61)
th_candidates = [0.35, 0.36, 0.37, 0.38, 0.39]

mu = args.mu
updates_per_epoch = args.updates_per_epoch


train_df = pd.read_csv(data_dir + 'train.csv')
print('Train : ', train_df.shape)
# test_df = pd.read_csv(data_dir + 'test.csv')
# print('Test : ', test_df.shape)

ema_n = int(train_df.shape[0] * (1 - 1 / n_folds) / (updates_per_epoch * batch_size))
print('ema_n:', ema_n)

print(args)

X = train_df['question_text']
y = train_df['target']

param_grid = {'fold': range(n_folds),
              'seed': split_seed_s}
scores = []
for params in ParameterGrid(param_grid):
    print(params)
    fold = params['fold']
    seed = params['seed']
    t1 = time.time()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, valid_idx = list(skf.split(X, y))[fold]
    tr_df = train_df.loc[train_idx]
    test_df = train_df.loc[valid_idx]
    tr_y = tr_df['target'].values
    test_y = test_df['target'].values

    obj = prepare_data(tr_df, test_df, maxlen, max_features,
                       trunc=trunc, lower=lower, clean_num=clean_num, unk_uni=unk_uni)
    tr_X_padded, test_X_padded, embedding_matrix, _ = obj
    tr_loader = get_dataloader(tr_X_padded, tr_y, batch_size=batch_size,
                               dtype_x=torch.long, dtype_y=torch.float, training=True,
                               drop_last=drop_last)
    test_loader = get_dataloader(test_X_padded, y=None, batch_size=batch_size,
                                 dtype_x=torch.long, training=False)
    print(f'Done Preprocessing:{time.time() - t1:.1f}s')

    test_pr_avg_s = [np.zeros((len(test_df), 1)) for _ in range(epochs)]
    for i in range(n_models):
        model_id = i + 1
        cols_in_use = cols_s[i % len(cols_s)]
        model = GRUModel(n_vocab, embed_dim, proj_dim, rnn_dim, n_layers, bidirectional, dense_dim,
                         pretrained_embedding=embedding_matrix[:, cols_in_use],
                         fix_embedding=fix_embedding, pme_relu=pme_relu, padding_idx=0)
        if i == 0:
            print(model)
            print('#Trainable params', get_param_size(model))
        model.cuda()
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        optimizer = Adam([p for n, p in model.named_parameters() if p.requires_grad is True])
        ema = EMA(model, mu, n=ema_n)
        for e in range(epochs):
            epoch = e + 1
            t2 = time.time()
            model.train()
            loss = run_epoch(model, tr_loader, optimizer, callbacks=[ema])
            ema.set_weights(ema_model)
            # To avoid RuntimeWarning:
            #   RNN module weights are not part of
            #   single contiguous chunk of memory. This means they need to
            #   be compacted at every call, possibly greatly increasing memory usage.
            #   To compact weights again call flatten_parameters().
            ema_model.gru.flatten_parameters()
            sc = params.copy()
            sc.update({'mu': mu, 'model_id': model_id, 'epoch': epoch})
            # val
            test_pr = ema_model.predict_proba(test_loader)
            test_pr_avg_s[e] += test_pr
            test_pr_avg = test_pr_avg_s[e] / model_id
            auc_one = roc_auc_score(test_y, test_pr)
            auc_avg = roc_auc_score(test_y, test_pr_avg)
            sc.update({'auc_one': auc_one, 'auc_avg': auc_avg})
            for th in th_candidates:
                f1_one, _ = f1_best(test_y, test_pr, [th])
                f1_avg, _ = f1_best(test_y, test_pr_avg, [th])
                sc.update({f'f1_one_{th:.2f}': f1_one, f'f1_avg_{th:.2f}': f1_avg})
            # F1 if we know the best threshold.
            f1_avg_best, th_avg_best = f1_best(test_y, test_pr_avg, thresh_s)
            sc.update({'f1_avg_best': f1_avg_best, 'th_avg_best': th_avg_best})
            scores.append(sc)
            print(f'model_id:{model_id} Epoch:{epoch} F1_best:{f1_avg_best:.4f} ' +
                  f'AVG AUC:{auc_avg:.4f} F1:{f1_avg:.4f} ' +
                  f'One AUC:{auc_one:.4f} F1:{f1_one:.4f} {time.time() - t2:.1f}s')

df = pd.DataFrame(scores)
df.to_csv(log_path)

pd.set_option('precision', 5)
pd.set_option('max_columns', 140)

df = df[df.epoch > 1]
keys = ['epoch']

print(args.exp_name)

grouped_avg = df[df.model_id == n_models].groupby(keys)
df_best = grouped_avg[['f1_avg_best']].mean()
df_best = df_best.join(grouped_avg[['th_avg_best']].mean())
df_best = df_best.join(grouped_avg[['auc_avg']].mean())
grouped_one = df.groupby(keys)
df_best = df_best.join(grouped_one[['auc_one']].mean())
print(df_best, '\n')

avg_s = []
one_s = []
for th in th_candidates:
    footer = f'_{th:.2f}'
    avg_s.append(df[df.model_id == n_models].groupby(keys)[['f1_avg' + footer]].mean())
    one_s.append(df.groupby(keys)[['f1_one' + footer]].mean())
print(pd.concat(avg_s, axis=1), '\n')
print(pd.concat(one_s, axis=1), '\n')

print(f'Done: {(time.time() - t0) / 3600:.1f}h')
