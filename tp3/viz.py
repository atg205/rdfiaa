from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base_path = Path(__file__).resolve().parent.parent
p = base_path / 'runs_new'
dfs = [pd.read_pickle(f) for f in p.glob('*.pckl')]
df = pd.concat(dfs, ignore_index=True)

def plot_group(base, group_vars, metrics, filename, xvar='epoch', huevar=None):
    grouped = base.groupby(group_vars)
    mean, std = grouped.mean().reset_index(), grouped.std().reset_index()

    for col, title in metrics:
        fig, ax = plt.subplots(figsize=(6,4))
        if huevar:
            for val in sorted(mean[huevar].unique()):
                m = mean[mean[huevar] == val]
                s = std[std[huevar] == val]
                ax.plot(m[xvar], m[col], '-o', label=f'{huevar}={val}')
                ax.fill_between(m[xvar], m[col]-s[col], m[col]+s[col], alpha=0.2)
        else:
            m, s = mean, std
            ax.plot(m[xvar], m[col], '-o')
            ax.fill_between(m[xvar], m[col]-s[col], m[col]+s[col], alpha=0.2)
        ax.set_xlabel(xvar)
        ax.set_ylabel(title)
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(base_path / 'plots_paper' / f'{filename}_{col}.pdf')
        plt.close()

metrics = [('train_acc','Train Accuracy'), ('test_top1','Test Accuracy'),
           ('trainloss','Train Loss'), ('testloss','Test Loss')]

# a) batch + lr
base = df.query('dropout_rate == 0 and not normalization and not crop and lr_decay == 1 and not batch_norm')
plot_group(base, ['batch_size','lr','epoch'], metrics, 'a_batch_lr', huevar='batch_size')

# b) normalization on/off
base = df.query('batch_size==128 and lr==0.1 and dropout_rate==0 and crop==False and lr_decay==1 and not batch_norm')
plot_group(base, ['normalization','epoch'], metrics, 'b_norm', huevar='normalization')

# c) crop on/off (normalized only)
base = df.query('batch_size==128 and lr==0.1 and dropout_rate==0 and normalization==True and lr_decay==1 and not batch_norm')
plot_group(base, ['crop','epoch'], metrics, 'c_crop', huevar='crop')

# d) lr decay variations
base = df.query('batch_size==128 and lr==0.1 and dropout_rate==0 and normalization==True and crop==True and not batch_norm')
plot_group(base, ['lr_decay','epoch'], metrics, 'd_lr_decay', huevar='lr_decay')

# e) dropout variations
base = df.query('batch_size==128 and lr==0.1 and normalization==True and crop==True and lr_decay==0.95 and not batch_norm')
plot_group(base, ['dropout_rate','epoch'], metrics, 'e_dropout', huevar='dropout_rate')

# f) batch norm on/off
base = df.query('batch_size==128 and lr==0.1 and dropout_rate==0.5 and normalization==True and crop==True and lr_decay==0.95')
plot_group(base, ['batch_norm','epoch'], metrics, 'f_batch_norm', huevar='batch_norm')
