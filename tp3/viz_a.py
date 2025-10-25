from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from viz import save_plots


# batch lr
filename = "a_batch_lr"

base_path = Path(__file__).resolve().parent.parent
p = base_path / 'runs_new'
dfs = [pd.read_pickle(f) for f in p.glob('*.pckl')]
df = pd.concat(dfs,ignore_index=True)
base = df.query('dropout_rate == 0 and not normalization and not crop and lr_decay == 1 and not batch_norm')

grouped = base.groupby(['batch_size','lr','epoch'])

metrics = [('train_acc','Train Accuracy'), ('test_top1','Test Accuracy'),
           ('trainloss','Train Loss'), ('testloss','Test Loss')]


mean = grouped.mean().reset_index()
std = grouped.std().reset_index()
for col, title in metrics:
    lrs = sorted(mean.lr.unique())
    fig, axes = plt.subplots(1, len(lrs), figsize=(10,4), sharey=True)
    for ax, lr in zip(axes, lrs):
        for b in sorted(mean.batch_size.unique()):
            m = mean[(mean.lr == lr) & (mean.batch_size == b)]
            s = std[(std.lr == lr) & (std.batch_size == b)]
            ax.plot(m.epoch, m[col], '-o', label=f'batch={b}')
            ax.fill_between(m.epoch, m[col]-s[col], m[col]+s[col], alpha=0.2)
        ax.set_title(f'{title} (lr={lr})')
        ax.set_xlabel('epoch')
        ax.legend()
    axes[0].set_ylabel(title)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(base_path / 'plots_paper' / f'{filename}{col}{title}.pdf')


# normalization
filename = "b_norm"
base = base = df.query('batch_size=128 and lr=0.1 and dropout_rate == 0 and normalization and not crop and lr_decay == 1 and not batch_norm')
# average and compare with not normalized

# crop 
base = base = df.query('batch_size=128 and lr=0.1 and dropout_rate == 0 and normalization and crop and lr_decay == 1 and not batch_norm')
# average and compare with previous (normalized)

# lr decay
base = base = df.query('batch_size=128 and lr=0.1 and dropout_rate == 0 and normalization and crop and not lr_decay == 1 and not batch_norm')
# here i want a visualisation over different lr decays (1 plot) with lr_decay = 0.8 0.9 and 0.95

# dropout
base = df.query('batch_size=128 and lr=0.1 and dropout_rate == 0 and normalization and crop and lr_decay == 0.95 and not batch_norm')
# here i want a visualisation over different lr decays (1 plot) with lr_decay = 0.3 0.4 and 0.5

# batch norm
base = df.query('batch_size=128 and lr=0.1 and dropout_rate == 0.5 and normalization and crop and lr_decay == 0.95 and batch_norm')
# here i want a visualisation of batch norm on and off 

# dropout
