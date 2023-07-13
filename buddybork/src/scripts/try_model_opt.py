import numpy as np
import matplotlib.pyplot as plt

from src.ml.ml_models import RecognitionModelLDA
from ossr_utils.misc_utils import get_colors, tags_str_to_int




np.random.seed(24352)

# settings
K = 3 # num of classes
N = [100] * K # samples per class
M = 1000 # num out-of-class samples
D_orig = 10 # original dimensionality
D_proj = 2 # LDA projected dimensionality

model_type = 'LDA'




# global offset
bias = 1000 * np.random.randn(D_orig)


# sample params
mu_lim = 10

mus = np.zeros((K, D_orig), dtype='float')
for k in range(K):
    mus[k, :] = np.random.rand(D_orig) * (2 * mu_lim) - mu_lim + bias

covs = []
for k in range(K):
    n_cov = int(1 * D_orig)
    cov_ = np.random.randn(n_cov, D_orig)
    cm = cov_ - np.mean(cov_, axis=0)
    cov_ = 1 / (n_cov - 1) * cm.T @ cm
    covs.append(5 * cov_)

# sample data
feats = []
tags = []
for k in range(K):
    feats.append(np.random.multivariate_normal(mus[k], covs[k], N[k]))
    tags.append([k] * N[k])
feats = np.vstack(feats)
tags = np.concatenate(tags)

out_lim = 20

feats_bg = np.random.randn(M, D_orig) * (2 * out_lim) + bias
tags_bg = np.array([K] * M)


# run opt
colors = get_colors(20, mult=1.0)[5:5 + (K + 1), :]
colors[-1, :] = np.array([0, 0, 0]) # outlier color

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)


if model_type == 'LDA':
    model = RecognitionModelLDA(D_proj_lda=D_proj)
tags_s = [str(tag_) for tag_ in tags]
tags_bg_s = [str(tag_) for tag_ in tags_bg]
if 1:
    feats_train = feats
    tags_train = tags_s
else:
    feats_train = np.vstack([feats, feats_bg])
    tags_train = tags_s + tags_bg_s
model.fit(feats_train, tags_train)


tags_all = [str(i) for i in range(K)]


def update(i=0):
    feats_proj = model._transformer.transform(feats_train)
    feats_bg_proj = model._transformer.transform(feats_bg)

    pred_feats = tags_str_to_int(model.predict(feats_train), K, tags_all=tags_all)
    colors_ = colors[pred_feats, :]
    pred_feats_bg = tags_str_to_int(model.predict(feats_bg), K, tags_all=tags_all)
    colors_bg_ = colors[pred_feats_bg, :]

    print(model.predict(feats_train))
    print(pred_feats)

    ax.cla()
    scat_feat = ax.scatter(feats_proj[:, 0], feats_proj[:, 1], c=colors_, s=10)
    scat_feat_bg = ax.scatter(feats_bg_proj[:, 0], feats_bg_proj[:, 1], c=colors_bg_, s=1)
    if model_type == 'LDA':
        for k in range(K):
            mus_k = model._classifier._cluster_params['mu'][k]
            ax.scatter(mus_k[:, 0], mus_k[:, 1], c='r', marker='+', s=50)
    ax.set_aspect('equal')

update()
plt.show()
