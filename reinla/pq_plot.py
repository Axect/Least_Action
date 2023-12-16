import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Import parquet file
df = pd.read_csv("./free_body_1d_bf.csv")

# Prepare Data to Plot
x = df['parameter_N'][:]
y = df['mean'][:]
p = np.polyfit(x, y, 3)
p = np.poly1d(p)
y_polyfit = p(np.array(x))

# Plot params
pparam = dict(
    xlabel = r'$N$ (the number of nodes)',
    ylabel = r'Time (sec)',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale()
    ax.set(**pparam)
    ax.plot(x, y, '.-', alpha=0.6, label=r'Brute-force $m=3$')
    ax.plot(x, y_polyfit, '--', alpha=0.6, label=r'Polyfit $\text{order}=3$')
    ax.legend()
    ax.grid()
    fig.savefig('free_body_1d_bf.png', dpi=600, bbox_inches='tight')
