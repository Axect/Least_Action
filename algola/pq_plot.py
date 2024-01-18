import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

df = pd.read_parquet('true.parquet')

t_true = np.array(df['t'][:])
q_true = np.array(df['q'][:])
t_start = t_true[0]
t_end = t_true[-1]
start = q_true[0]
end = q_true[-1]

dh = pd.read_parquet('data.parquet')

species = 3

t_hats = [np.linspace(t_start, t_end, 2**i + 1) for i in range(1, species+1)]
bfs = []
dcs = []
for i in range(1, 4):
    bf = np.array(dh[f'bf_{i}'][:2**i-1])
    dc = np.array(dh[f'dc_{i}'][:2**i-1])
    bf = np.insert(bf, 0, start)
    bf = np.append(bf, end)
    dc = np.insert(dc, 0, start)
    dc = np.append(dc, end)
    bfs.append(bf)
    dcs.append(dc)
styles = ['-', '--', ':']

pparam = dict(
    xlabel = r'$t$',
    ylabel = r'$q$',
)

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(t_true, q_true, 'k', label=r'$q_\text{true}~(\omega=1, T=\frac{2}{3}\pi)$', alpha=0.3)
    for i in range(1, 4):
        ax.plot(t_hats[i-1], bfs[i-1], styles[i-1], label=rf'$\hat{{q}}_\text{{bf}} ~(N=2^{i} - 1)$', alpha=0.7)
    ax.legend()
    fig.savefig('harmonic_bf.png', dpi=600, bbox_inches='tight')

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(t_true, q_true, 'k', label=r'$q_\text{true}~(\omega=1, T=\frac{2}{3}\pi)$', alpha=0.3)
    for i in range(1, 4):
        ax.plot(t_hats[i-1], dcs[i-1], styles[i-1], label=rf'$\hat{{q}}_\text{{dc}} ~(N=2^{i} - 1)$', alpha=0.7)
    ax.legend()
    fig.savefig('harmonic_dc.png', dpi=600, bbox_inches='tight')
