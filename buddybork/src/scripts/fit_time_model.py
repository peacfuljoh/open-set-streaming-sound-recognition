import numpy as np
import matplotlib.pyplot as plt

from src.utils.feats_io_utils import get_annots_df_for_days
from src.ml.classifiers import fit_time_model
from ossr_utils.misc_utils import get_colors, get_times_from_dts
from src.constants_ml import TM_GRID

# get annots
days = [['2022-01-01', '2024-01-01']] # inclusive
tags_db = None
# tags_db = ['bork', 'bwrph', 'kibble', 'door_close', 'keys', 'crate', 'doorbell']

annots = get_annots_df_for_days(days)
if tags_db is not None:
    annots = annots[np.any(annots['tag'].to_numpy() == np.array(tags_db)[:, np.newaxis], axis=0)]
print(annots)
tags_all = annots['tag'].to_numpy()
tags = np.unique(tags_all)

num_tags = len(tags)
num_annots = len(annots)


# get time-of-day from datetime
times = get_times_from_dts(list(annots['datetime_start']))


# fit KDEs
t_probs = np.zeros((len(TM_GRID), num_tags), dtype='float')
for i, tag in enumerate(tags):
    t_probs[:, i] = fit_time_model(times[tags_all == tag])


# plot
colors = get_colors(num_tags, mult=0.8)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

for i, tag in enumerate(tags):
    idxs = tags_all == tag
    annots_i = annots[idxs]
    num_i = len(annots_i)

    ax.scatter(times[idxs], i * np.ones(num_i), color=colors[i, :], s=2, zorder=10)
    ax.plot(TM_GRID, 0.8 * t_probs[:, i] + i, c=colors[i, :], zorder=5)
    ax.plot(TM_GRID[[0, -1]], i * np.ones(2), c='gray', linestyle='--', alpha=0.2, zorder=0)

ax.set_xlim([0, 24])
ax.set_title('Annotation distributions by time of day')
ax.set_yticks(np.arange(num_tags))
ax.set_yticklabels(tags)

plt.show()
