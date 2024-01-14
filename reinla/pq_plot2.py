import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Import parquet file
df = pd.read_csv("./fb_1d_dc.csv")

# Prepare Data to Plot
x = np.array([(2**(20) - 1) * i for i in range(1, 11)])
y = df['mean'][:]
p = np.polyfit(x, y, 1)
p = np.poly1d(p)
y_polyfit = p(np.array(x))

# Plot params
pparam = dict(
    xlabel = r'$N$ (the number of nodes)',
    ylabel = r'Time (sec)',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots(figsize=(4, 2.25))
    ax.autoscale()
    ax.set(**pparam)
    ax.plot(x, y, '.-', alpha=0.6, label=r'Divide and Conquer $m=7$')
    ax.plot(x, y_polyfit, '--', alpha=0.6, label=r'Polyfit $\text{order}=1$')
    ax.legend()
    ax.grid()
    fig.savefig('free_body_1d_dnc.png', dpi=600, bbox_inches='tight')
