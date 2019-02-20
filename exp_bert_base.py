import time
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import random

import torch
from torch.nn import BCEWithLogitsLoss
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

import os
from utility_bert import QuoraProcessor, convert_examples_to_features, get_dataloader
from utility import f1_best, disp_elapsed
from utility_pytorch import get_param_size
from utility_bert import BertCl

t0 = time.time()

data_dir = '/home/tks/data/Quora2/'

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name',
                    default='', type=str, required=True)

parser.add_argument('--exp_num',
                    default=999, type=int, required=False,
                    help='0-999: Experiment number. The log filename is exp{exp_num}.csv')

parser.add_argument('--n_folds',
                    default=5, type=int, required=False,
                    help='Number of CV folds.')
parser.add_argument('--fold',
                    default=0, type=int, required=False,
                    help='Which fold to try.')
parser.add_argument('--seed_split',
                    default=100, type=int, required=False,
                    help='Seed for data split.')


parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, ")
parser.add_argument("--task_name",
                    default="Quora",
                    type=str,
                    required=False,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default="./",
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")

# Other parameters
parser.add_argument("--max_seq_length",
                    default=50,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                    "Sequences longer than this will be truncated, and sequences shorter \n"
                    "than this will be padded.")
parser.add_argument("--do_lower_case",
                    default=0,
                    type=int)
parser.add_argument("--do_train",
                    default=True,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=False,
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                    "E.g., 0.1 = 10%% of training.")

parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=8,
                    help="Number of updates steps to accumualte before performing a backward/update pass.")
parser.add_argument('--optimize_on_cpu',
                    default=False,
                    action='store_true',
                    help="Whether to perform optimization and keep the optimizer averages on CPU")
parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type=float, default=128,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

parser.add_argument("--weight_0",
                    default=1.0,
                    type=float,
                    help="a weight of negative examples.")
parser.add_argument("--weight_1",
                    default=1.0,
                    type=float,
                    help="a weight of positive examples.")

parser.add_argument("--no_weight_decay",
                    default=0,
                    type=float)
parser.add_argument("--stratify",
                    default=1,
                    type=int)

# Model
parser.add_argument('--n_bertlayers',
                    default=1, type=int, required=False)

# Training
parser.add_argument('--n_dev',
                    default=100000, type=int, required=False,
                    help='The size of developping set.')

parser.add_argument('--n_models',
                    default=1, type=int, required=False)


# DIR
parser.add_argument('--log_dir',
                    default='log/', type=str, required=False)

args = parser.parse_args()

with open(args.log_dir + f'exp{args.exp_num:03d}.json', 'w') as f:
    json.dump(vars(args), f, indent=4)


device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

exp_num = args.exp_num

log_fname = f'exp{exp_num:03d}.csv'
log_path = args.log_dir + log_fname

seed = args.seed

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


seed_split = args.seed_split
n_folds = args.n_folds
fold = args.fold
n_dev = args.n_dev

maxlen = args.max_seq_length

modelname = args.bert_model
n_bertlayers = args.n_bertlayers

n_models = args.n_models

args.eval_batch_size = 2 * args.train_batch_size

if args.weight_0 != args.weight_1:
    args.pos_weight = torch.Tensor([args.weight_1 / args.weight_0]).to(device)
else:
    args.pos_weight = None

print(args)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(model, data_loader):
    preds = []
    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, _ = batch
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            preds.append(logits.detach().cpu().numpy())
    preds = np.concatenate(preds) if len(preds) > 1 else preds[0]
    return preds[:, 0]


def predict_proba(model, data_loader):
    return sigmoid(predict(model, data_loader))


def run_epoch(model, dataloader, optimizer, criterion, args, callbacks=None):
    """
    callbacks: [func, ..] on_batch_end, on_epoch_end
    """
    gradient_accumulation_steps = args.gradient_accumulation_steps
    t1 = time.time()
    tr_loss = 0
    for step, batch in enumerate(tr_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        label_ids = label_ids.float()
        outputs = model(input_ids, segment_ids, input_mask)
        loss = criterion(outputs[:, 0], label_ids)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        tr_loss += loss.item()
        if (step + 1) % 2000 == 0:
            loss_now = gradient_accumulation_steps * tr_loss / (step + 1)
            print(f'step:{step+1} loss:{loss_now:.7f} time:{time.time() - t1:.1f}s')
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
            if callbacks is not None:
                for func in callbacks:
                    func.on_batch_end(model)
    if callbacks is not None:
        for func in callbacks:
            func.on_epoch_end(model)
    return gradient_accumulation_steps * tr_loss / (step + 1)


def get_bert_cl(modelname, tr_size, args,
                n_bertlayers, dropout=0.1, num_labels=1):
    model = BertCl(modelname, n_bertlayers, dropout, num_labels)
    model.to(device)

    param_optimizer = list(model.named_parameters())

    if args.no_weight_decay == 1:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer], 'weight_decay_rate': 0.0}

        ]
    else:
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    num_train_steps = int(
        tr_size / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    return model, optimizer


# Preprocessing begin
train_df = pd.read_csv(data_dir + 'train.csv')
print('Train shape : ', train_df.shape)


skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_split)
X = train_df['question_text']
y = train_df['target']

train_idx, valid_idx = list(skf.split(X, y))[fold]
stratify = y[train_idx] if args.stratify == 1 else None
tr_df, dev_df = train_test_split(train_df.loc[train_idx],
                                 test_size=n_dev/train_idx.size,
                                 stratify=stratify,
                                 random_state=seed_split)

tr_df = tr_df.reset_index(drop=True)
dev_df = dev_df.reset_index(drop=True)
val_df = train_df.loc[valid_idx]


# Load pre-trained model tokenizer (vocabulary)
def get_loader_s(modelname):
    do_lower_case = modelname.split('-')[-1] == 'uncased' or args.do_lower_case > 0
    tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=do_lower_case)

    processor = QuoraProcessor()
    tr_examples = processor._create_examples(tr_df)
    dev_examples = processor._create_examples(dev_df)
    val_examples = processor._create_examples(val_df)

    tr_features = convert_examples_to_features(tr_examples, maxlen, tokenizer, ['0', '1'])
    dev_features = convert_examples_to_features(dev_examples, maxlen, tokenizer, ['0', '1'])
    val_features = convert_examples_to_features(val_examples, maxlen, tokenizer, ['0', '1'])
    n_tr = len(tr_features)

    tr_dataloader = get_dataloader(tr_features, batch_size=args.train_batch_size)
    dev_dataloader = get_dataloader(dev_features, training=False, batch_size=args.eval_batch_size)
    val_dataloader = get_dataloader(val_features, training=False, batch_size=args.eval_batch_size)

    return tr_dataloader, dev_dataloader, val_dataloader, n_tr


print('\nEffective batch_size:', args.train_batch_size * args.gradient_accumulation_steps)


tr_dataloader, dev_dataloader, val_dataloader, n_tr = get_loader_s(modelname)
print(f'Done preprocessing:{time.time() - t0:.1f}s')
pr_dev_avg_s = [np.zeros(len(dev_df)) for _ in range(int(args.num_train_epochs))]
pr_val_avg_s = [np.zeros(len(val_df)) for _ in range(int(args.num_train_epochs))]

criterion = BCEWithLogitsLoss(pos_weight=args.pos_weight)
scores = []
for j in range(n_models):
    n_model = j + 1
    model, optimizer = get_bert_cl(modelname, n_tr, args,
                                   n_bertlayers=n_bertlayers)
    if j == 0:
        print('#Params:', get_param_size(model))
    for i in range(int(args.num_train_epochs)):
        t1 = time.time()
        epoch = i + 1
        print(f'Epoch:{epoch}')
        model.train()
        loss = run_epoch(model, tr_dataloader, optimizer, criterion, args)
        # Single
        model.eval()
        pr_dev = predict_proba(model, dev_dataloader)
        pr_val = predict_proba(model, val_dataloader)
        _, th_one = f1_best(dev_df['target'], pr_dev)
        f1_one, _ = f1_best(val_df['target'], pr_val, thresh_s=[th_one])
        f1_ub, th_ub = f1_best(val_df['target'], pr_val)
        auc_one = roc_auc_score(val_df['target'], pr_val)
        # Average ensemble
        pr_dev_avg_s[i] += pr_dev
        pr_val_avg_s[i] += pr_val
        pr_dev_avg = pr_dev_avg_s[i] / n_model
        pr_val_avg = pr_val_avg_s[i] / n_model
        _, th_avg = f1_best(dev_df['target'], pr_dev_avg)
        f1_avg, _ = f1_best(val_df['target'], pr_val_avg, thresh_s=[th_avg])
        auc_avg = roc_auc_score(val_df['target'], pr_val_avg)
        tm_epoch = time.time() - t1
        print(f'  F1:{f1_one:.5f} threshold:{th_one:.2f} AUC:{auc_one:.5f} time:{tm_epoch:.1f}s')
        scores.append({'n_model': n_model, 'epoch': epoch,
                       'f1_one': f1_one, 'auc_one': auc_one, 'th_one': th_one,
                       'f1_avg': f1_avg, 'auc_avg': auc_avg, 'th_avg': th_avg,
                       'f1_ub': f1_ub, 'th_ub': th_ub,
                       'time': tm_epoch})


df = pd.DataFrame(scores)
df.to_csv(log_path, index=False)

pd.set_option('precision', 5)
pd.set_option('max_columns', 120)

keys = ['epoch']

grouped_one = df.groupby(keys)
df_one = grouped_one[['f1_one', 'auc_one', 'th_one']].mean()
df_one.columns = ['F1', 'AUC', 'threshold']
print('\nSingle')
print(df_one)

if n_models > 1:
    df_avg = df[df.n_model == n_models].groupby(keys)[['f1_avg', 'auc_avg']].mean()
    df_avg.columns = ['F1', 'AUC']
    print(f'\nAvg ensemble of {n_models:d} models')
    print(df_avg, '\n')

print(f'\n{df["time"].mean():.1f}s/epoch')

disp_elapsed(t0)
