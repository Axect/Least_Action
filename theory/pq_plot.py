import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# Import parquet file
df = pd.read_parquet('data.parquet')

# Prepare Data to Plot
x = df['x']
y_harmonic = df['harmonic']
y_pade = df['pade']
y_taylor = df['taylor']
y_one = df['one']
y_three = df['three']


# Plot params
pparam = dict(
    xlabel = r'$\omega T$',
    ylabel = r'$q$',
    xscale = 'linear',
    yscale = 'linear',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, y_harmonic, 'k', label=r'$q_\text{true}(T/2)$', alpha=0.3)
    ax.plot(x, y_pade, '--', label=r'$q_\text{Padé}(T/2)$', alpha=0.7)
    ax.plot(x, y_taylor, '-.', label=r'$q_\text{Taylor}(T/2)$', alpha=0.7)
    ax.plot(x, y_one, ':', label=r'$q_1(N=1)$', alpha=0.7)
    ax.plot(x, y_three, '-.', label=r'$q_2(N=3)$', alpha=0.7)
    ax.legend()
    fig.savefig('plot.png', dpi=600, bbox_inches='tight')

# Plot 2
dg = pd.read_parquet('data2.parquet')

k = dg['k']
y_hat = dg['y_hat']
y_true = dg['y_true']

y_diff = np.abs(np.array(y_true) - np.array(y_hat)) / np.array(y_true)

pparam = dict(
    xlabel = r'$k$',
    ylabel = r'Central node',
    ylim = (np.min(y_hat)*0.999, np.max(y_true) * 1.001),
)

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(k, y_hat, 'k.-', label=r'$q_k$')
    ax.plot(k, y_true, 'r--', label=r'$q_\text{true}(T/2) (\omega T = \dfrac{\pi}{2})$')
    ax.legend()
    fig.savefig('plot2.png', dpi=600, bbox_inches='tight')

pparam = dict(
    xlabel = r'$k$',
    ylabel = r'Error of central node (\%)',
    ylim = (-0.1, 5),
)

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(k, y_diff * 100, 'k.-', label=r'$\frac{|q_k - q_\text{true}(T/2)|}{q_\text{true}(T/2)} \times 100$')
    ax.axhline(y=0, color='r', linestyle='--', label=r'$0$')
    ax.legend()
    fig.savefig('plot3.png', dpi=600, bbox_inches='tight')

# Plot 3
dh = pd.read_parquet('data3.parquet')

t = dh['t']
y_true = dh['y_true']
t_hats = [np.linspace(0, np.max(t), 2**i + 1) for i in range(1, 5)]
y_hats = []
for i in range(1, 5):
    y_hats.append(dh[f'y_{i}'][:len(t_hats[i-1])])
styles = ['-', '--', '-.', ':']

pparam = dict(
    xlabel = r'$t$',
    ylabel = r'$q$',
)

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(t, y_true, 'k', label=r'$q_\text{true}~(\omega=1, T=\frac{2}{3}\pi)$', alpha=0.3)
    for i in range(1, 5):
        ax.plot(t_hats[i-1], y_hats[i-1], styles[i-1], label=rf'$\hat{{q}} ~(N=2^{i} - 1)$', alpha=0.5)
    ax.legend()
    fig.savefig('plot4.png', dpi=600, bbox_inches='tight')
