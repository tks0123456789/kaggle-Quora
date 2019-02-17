import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--exp_num_s',
                    default='', type=str, required=True,
                    help='String of space delimited experiment numbers to plot.')
parser.add_argument('--log_dir',
                    default='log/', type=str, required=False)
parser.add_argument('--img_dir',
                    default='./', type=str, required=False)


parser.add_argument('--width',
                    default=10, type=int, required=False)
parser.add_argument('--height',
                    default=10, type=int, required=False)

parser.add_argument('--metric',
                    default='F1', type=str, required=False)
parser.add_argument('--min_epoch',
                    default=2, type=int, required=False)
parser.add_argument('--max_epoch',
                    default=10, type=int, required=False)
parser.add_argument('--threshold',
                    default=0.36, type=float, required=False)

args = parser.parse_args()
print(args)

exp_num_s = [int(val) for val in args.exp_num_s.split()]

config_s = []
for exp_num in exp_num_s:
    with open(args.log_dir + f'exp{exp_num:03d}.json', 'r') as f:
        config_s.append(json.load(f))

df_scores_s = [pd.read_csv(args.log_dir + f'exp{exp_num:03d}.csv') for exp_num in exp_num_s]

if args.metric == 'F1':
    avg_col = f'f1_avg_{args.threshold:.02f}'
    one_col = f'f1_one_{args.threshold:.02f}'
elif args.metric == 'AUC':
    avg_col = 'auc_avg'
    one_col = 'auc_one'

sc_one_s = []
sc_avg_s = []
for df_scores, config in zip(df_scores_s, config_s):
    df = df_scores[(df_scores['epoch'] >= args.min_epoch) & (df_scores['epoch'] <= args.max_epoch)]
    sc_avg_s.append(df[df.model_id == config['n_models']].groupby('epoch')[avg_col].mean())
    sc_avg_s[-1].name = config['exp_name']
    sc_one_s.append(df.groupby('epoch')[one_col].mean())
    sc_one_s[-1].name = config['exp_name']

df_one = pd.concat(sc_one_s, axis=1)
df_avg = pd.concat(sc_avg_s, axis=1)

title = args.metric
if args.metric == 'F1':
    title += f', threshold={args.threshold:.02f}'

fig, axes = plt.subplots(2, 1, figsize=(args.width, args.height))

fig.suptitle(title)

df_avg.plot(ax=axes[0])
axes[0].set_title('Average Ensemble')
axes[0].set_xticks([])
axes[0].xaxis.set_label_text('')

df_one.plot(ax=axes[1])
axes[1].set_title('Single')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.92)

imgname = args.metric + '_' + '__'.join([config['exp_name'] for config in config_s]) + '.png'
plt.savefig(args.img_dir + imgname)
