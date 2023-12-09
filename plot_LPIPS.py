import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
fig = plt.figure(figsize=(6,3.5))
# sinusoidal lines with colors from default color cycle
L = 2*np.pi
# group1 [0.16, 0.22, 0.34, 0.37, 0.45 ,0.47]
# group2 [0.15, 0.23, 0.31, 0.41, 0.42, 0.46]
x = [1,2,4,8,16,32]
ncolors = len(plt.rcParams['axes.prop_cycle'])
y = [0.15,0.22,0.31,0.37,0.42,0.46]

plt.plot(x, y, '-o')
# plt.xlabel('batch size', fontsize=14)
# plt.ylabel('LPIPS', fontsize=14)
ax = plt.gca()
ax.xaxis.set_ticks(x)
bwith = 2
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
# ax.label_outer()
plt.tick_params(labelsize=12)
plt.grid( color = 'black',linestyle='-.',linewidth = 1)
# fig.autofmt_xdate()
# plt.savefig('LPIPS-BatchSize.eps', format='eps')
plt.show()
