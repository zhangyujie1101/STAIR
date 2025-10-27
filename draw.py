import matplotlib.pyplot as plt
import numpy as np

gamma = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
recall = np.array([0.1016, 0.1038, 0.1043, 0.1037, 0.1013, 0.1005, 0.0997])
ndcg = np.array([0.0439, 0.0447, 0.0456, 0.0454, 0.0447, 0.0443, 0.0438])

fig, ax1 = plt.subplots(figsize=(4, 3))

# 左轴 Recall@20
color1 = 'gray'
ax1.plot(gamma, recall, 'o-', color=color1, label='Recall@20', linewidth=2)
ax1.set_xlabel(r'$\gamma$', fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.090, 0.105)

# 在最佳点画星形标记
best_idx_recall = np.argmax(recall)
ax1.plot(gamma[best_idx_recall], recall[best_idx_recall],
         marker='*', color=color1, markersize=10)

# 右轴 NDCG@20
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.plot(gamma, ndcg, 'o-', color=color2, label='NDCG@20', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.040, 0.050)

# 在最佳点画星形标记
best_idx_ndcg = np.argmax(ndcg)
ax2.plot(gamma[best_idx_ndcg], ndcg[best_idx_ndcg],
         marker='*', color=color2, markersize=10)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='lower left', frameon=False)

plt.tight_layout()
plt.savefig('γ折线图.png')
plt.show()
