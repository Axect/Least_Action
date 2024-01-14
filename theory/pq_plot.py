import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Import parquet file
df = pd.read_parquet('data.parquet')

# Prepare Data to Plot
x = df['x']
y_harmonic = df['harmonic']
y_pade = df['pade']
y_taylor = df['taylor']
y_one = df['one']


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
    ax.plot(x, y_pade, '--', label=r'$q_\text{pade}(T/2)$', alpha=0.7)
    ax.plot(x, y_taylor, '-.', label=r'$q_\text{taylor}(T/2)$', alpha=0.7)
    ax.plot(x, y_one, ':', label=r'$q_1$', alpha=0.7)
    ax.legend()
    fig.savefig('plot.png', dpi=600, bbox_inches='tight')
