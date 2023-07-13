
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if 0:
    from ossr_utils.audio_utils import read_wav
    from ossr_utils.fft_utils import stft
    from src.constants_ml import WIN_SIZE, HOP_SIZE, SAMPLERATE
    from src.constants_stream import MAX_VAL_RAW_AUDIO
    from src.ml.transformers import TransformerLDA, TransformerPCA
    from src.ml.classifiers import ClassifierGMM
    from ossr_utils.misc_utils import logsumexp


fontsize = 15


"""Plot waveform and STFT"""
if 0:
    # get audio
    wav_fname = "C:/Users/Johtr/Desktop/borkfest_2023-01-29.wav"
    sr, wf = read_wav(wav_fname)
    t_lims = [10, 25]
    wf = wf[int(t_lims[0] * sr):int(t_lims[1] * sr)]

    # compute spec
    spec_all = np.abs(stft(wf, WIN_SIZE, HOP_SIZE)) # num_freqs x num_frames
    max_amp = np.max(np.abs(wf))

    # plot
    fig = plt.figure(figsize=(12, 8))

    ax0 = fig.add_subplot(2, 1, 1)
    ax0.plot(t_lims[0] + np.arange(len(wf)) / SAMPLERATE, wf / MAX_VAL_RAW_AUDIO, c='k')
    ax0.set_title('Waveform', fontsize=fontsize)
    # ax0.set_xlabel('Time (s)', fontsize=fontsize)
    ax0.set_ylabel('Amplitude', fontsize=fontsize)
    ax0.tick_params(axis='x', labelsize=fontsize)
    ax0.tick_params(axis='y', labelsize=fontsize)

    ax1 = fig.add_subplot(2, 1, 2, sharex=ax0)
    ax1.imshow(spec_all ** 0.4, origin='lower', interpolation='none', aspect='auto',
               cmap=plt.cm.get_cmap('jet'),
               extent=[t_lims[0], t_lims[0] + len(wf) / SAMPLERATE, 0, SAMPLERATE / 2])
    ax1.set_ylim([0, 10000])
    ax1.set_title('Spectrogram', fontsize=fontsize)
    ax1.set_xlabel('Time (s)', fontsize=fontsize)
    ax1.set_ylabel('Frequency (Hz)', fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)

    plt.show()




"""PCA"""
if 0:
    np.random.seed(1)

    # make data
    N = 200
    mu = np.array([30, 60, 40])
    N_c = 5
    cov = np.random.randn(N_c, 3)
    cov = 1 / (N_c - 1) * cov.T @ cov
    X = np.random.multivariate_normal(mu, cov, N)

    # get projector and apply it
    tr = TransformerPCA(3, 2)
    tr.fit(X)
    A_pca = tr._A * np.sqrt(N)
    mu_x = np.mean(X, axis=0)
    # u, s, vh = np.linalg.svd(X - mu_x, full_matrices=False)
    # A_pca = vh[:2, :].T * (1 / s[:2]) * np.sqrt(N)
    Xp = (X - mu_x) @ A_pca

    cov_xp = 1 / (N - 1) * Xp.T @ Xp
    print(cov_xp)

    # plot
    fig = plt.figure(figsize=(8, 5))

    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', s=5)
    ax0.set_title('Original data')
    ax0.set_aspect('equal')

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.scatter(Xp[:, 0], Xp[:, 1], c='b', s=5)
    ax1.set_title('Centered and projected data')
    ax1.set_aspect('equal')

    plt.show()

"""LDA"""
if 0:
    np.random.seed(1)

    # make data
    K = 3
    N = 200
    mus = np.array([[30, 60, 40],
                    [60, 20, 10],
                    [50, 30, 60]])
    N_c = 5
    covs = []
    for k in range(K):
        cov_k = 10 * np.random.randn(N_c, 3)
        cov_k = 1 / (N_c - 1) * cov_k.T @ cov_k
        covs.append(cov_k)
    X = np.zeros((N, 3), dtype='float')
    z = np.zeros(N)
    for i in range(N):
        k = np.random.randint(K)
        z[i] = k
        X[i, :] = np.random.multivariate_normal(mus[k, :], covs[k], 1)

    # get projector and apply it
    tr = TransformerLDA(3, 3, 2, K)
    A_lda = tr._calc_LDA_projector(X, z)
    mu_x = np.mean(X, axis=0)
    Xp = (X - mu_x) @ A_lda

    # plot
    colors = ['r', 'g', 'b']

    fig = plt.figure(figsize=(8, 5))

    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    for k in range(K):
        idxs = z == k
        ax0.scatter(X[idxs, 0], X[idxs, 1], X[idxs, 2], c=colors[k], s=5)
    ax0.set_title('Original data')
    ax0.set_aspect('equal')

    ax1 = fig.add_subplot(1, 2, 2)
    for k in range(K):
        idxs = z == k
        ax1.scatter(Xp[idxs, 0], Xp[idxs, 1], c=colors[k], s=5)
    ax1.set_title('LDA-transformed data')
    ax1.set_aspect('equal')

    plt.show()




"""GMMs"""
if 0:
    np.random.seed(1)

    # make data
    Nt = 2 # num tags
    K = 3 # num clusters per tag
    N = 200 # num samps per tag
    mus = np.array([[20, 60],
                    [30, 75],
                    [30, 55],
                    [70, 35],
                    [55, 55],
                    [55, 20]])
    N_c = 4
    covs = []
    for k in range(K * Nt):
        cov_k = 8 * np.random.randn(N_c, 2)
        cov_k = 1 / (N_c - 1) * cov_k.T @ cov_k
        covs.append(cov_k)

    N_tot = N * Nt
    X = np.zeros((N_tot, 2), dtype='float')
    z = np.zeros(N_tot, dtype='int') # overall cluster idx
    zt = np.zeros(N_tot, dtype='int') # class index
    for i in range(N_tot):
        t_i = np.random.randint(Nt)
        k_i = np.random.randint(K)
        zt[i] = t_i
        z[i] = t_i * K + k_i
        X[i, :] = np.random.multivariate_normal(mus[z[i], :], covs[z[i]], 1)

    # fit GMMs
    Nc = 10
    cl = ClassifierGMM(Nt, num_clusters=Nc)
    cl.fit(X, zt)
    mus_ = cl._cluster_params['mu']
    vars = cl._cluster_params['var']
    weights = cl._cluster_params['w']

    # plot prob-colored data w/ clusters
    colors = ['b', 'r']
    colors_rgb = np.array([[0, 0, 1], [1, 0, 0]])
    mix = 0.75
    colors_rgb = mix * colors_rgb + (1 - mix) * np.ones(3)

    fig = plt.figure(figsize=(8, 5))

    ax0 = fig.add_subplot(1, 2, 1)
    for t in range(Nt):
        idxs = zt == t
        ax0.scatter(X[idxs, 0], X[idxs, 1], c=colors_rgb[t], s=5)
    ax0.set_title('Original data')
    ax0.set_aspect('equal')

    ax1 = fig.add_subplot(1, 2, 2, sharex=ax0, sharey=ax0)
    angs = np.linspace(0, 2 * np.pi, 100)
    xy = np.vstack((np.cos(angs), np.sin(angs)))
    for t in range(Nt):
        for k in range(Nc):
            mu_k = mus_[t][k]
            A = np.eye(2) * np.sqrt(vars[t][k])
            circ = A @ xy
            wm = np.min(weights[t])
            alpha_k = (weights[t][k] - wm) / (np.max(weights[t]) - wm) # in [0, 1]
            alpha_k = 0.3 + alpha_k * 0.7
            ax1.plot(mu_k[0] + circ[0, :], mu_k[1] + circ[1, :], c=0.65 * colors_rgb[t, :],
                     alpha=alpha_k)
    metric = -cl.get_metric(X) # log likelihood
    probs = np.exp(metric - logsumexp(metric, axis=1)) # pred probs
    colors_x = probs @ colors_rgb # blended colors based on pred probs
    ax1.scatter(X[:, 0], X[:, 1], c=colors_x, s=5)
    ax1.set_title('Clusters and data probs')
    ax1.set_aspect('equal')


    # plot prob-colored data with detection contours
    xmin = np.min(X, axis=0) - 50
    xmax = np.max(X, axis=0) + 50
    N_bg = 100
    X_bg = np.random.rand(N_bg, 2) * (xmax - xmin) + xmin
    X_w_bg = np.vstack((X, X_bg))
    metric = -cl.get_metric(X_w_bg)  # log likelihood
    probs = np.exp(metric - logsumexp(metric, axis=1))  # pred probs
    colors_x = probs @ colors_rgb  # blended colors based on pred probs
    colors_x = np.maximum(np.minimum(colors_x, 1), 0)
    colors_x[np.all(metric < -10, axis=1), :] = 0

    Ng = 200
    xx, yy = np.meshgrid(np.linspace(xmin[0], xmax[0], Ng), np.linspace(xmin[1], xmax[1], Ng))
    xy = np.vstack((xx.ravel(), yy.ravel())).T

    metrics = []
    for jj in [[0], [1], [0, 1]]:
        # https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/pcolormesh_levels.html#sphx-glr-gallery-images-contours-and-fields-pcolormesh-levels-py
        metric_xy = np.max(-cl.get_metric(xy)[:, jj], axis=1)
        metric_xy = metric_xy.reshape(Ng, Ng)
        metric_xy = metric_xy[:-1, :-1]
        metrics.append(metric_xy)
    metrics_min = metrics[2].min()
    for metric_ in metrics:
        metric_ -= metrics_min
    metrics_min = metrics[2].min()
    metrics_max = metrics[2].max()

    for j in range(3):
        fig = plt.figure(figsize=(8, 5))

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(X_w_bg[:-N_bg, 0], X_w_bg[:-N_bg, 1], c=colors_x[:-N_bg, :], s=5, marker='o', zorder=5)
        ax1.scatter(X_w_bg[-N_bg:, 0], X_w_bg[-N_bg:, 1], c=colors_x[-N_bg:, :], s=5, marker='^', zorder=5)
        ax1.set_title('Detection metric contours')
        ax1.set_aspect('equal')

        # levels = MaxNLocator(nbins=15).tick_values(metrics[j].min(), metrics[j].max())
        levels = MaxNLocator(nbins=15).tick_values(metrics_min, metrics_max)
        ax1.contourf(xx[:-1, :-1], yy[:-1, :-1], metrics[j], levels=levels, cmap=plt.get_cmap('plasma'), zorder=0)
        ax1.set_xlim(xmin[0], xmax[0])
        ax1.set_ylim(xmin[1], xmax[1])

    plt.show()



"""Confusion matrix results"""
if 1:
    mat = np.array([
        [123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 12, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 17, 0, 0, 0, 0, 0, 2, 2, 1, 0, 2],
        [1, 0, 0, 2, 109, 1, 0, 0, 0, 18, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 8, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
        [1, 27, 0, 0, 11, 1, 6, 3, 3, 978, 30, 11, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    ])
    tags_train = np.array(['bork', 'bwrph', 'crate', 'dabadab', 'door_close', 'keys', 'kibble', 'other', 'sneeze', 'snuck',
                  'tags_jingle'])
    tags_test = np.array(['bork', 'bwrph', 'crate', 'dabadab', 'door_close', 'doorbell', 'growl', 'keys', 'kibble', 'other',
                 'pots_and_pans', 'sneeze', 'snuck', 'tags_jingle'])
    prec = []
    recall = []
    for tag in tags_train:
        i = np.where(tags_train == tag)[0][0]
        j = np.where(tags_test == tag)[0][0]
        prec_i = mat[i, j] / np.sum(mat[i, :])
        recall_i = mat[i, j] / np.sum(mat[:, j])
        prec.append(prec_i)
        recall.append(recall_i)
    print(np.round((100 * np.vstack((prec, recall)).T)).astype('int'))
