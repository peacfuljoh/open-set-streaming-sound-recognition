'''
Featurization of waveforms
'''

from typing import Tuple, Optional, List

import numpy as np

from src.constants_ml import WIN_SIZE, HOP_SIZE, MAX_NUM_SPEC_FEATS
from ossr_utils.fft_utils import stft
from src.constants_stream import SAMPLERATE


def featurize(wf: np.ndarray,
              samp_start: int,
              samp_end: int,
              tag: Optional[str] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from a waveform"""
    assert 0 <= samp_start < len(wf)
    assert 0 <= samp_end < len(wf)

    wf_annot = wf[samp_start:samp_end]

    # stft
    spec = np.abs(stft(wf_annot, WIN_SIZE, HOP_SIZE)) # num_freqs x num_frames
    N = spec.shape[1]

    # select loudest frames
    idxs_sort = np.argsort(np.sum(spec, axis=0))[::-1]
    num_frames = min(N, MAX_NUM_SPEC_FEATS)
    idxs_max = idxs_sort[:num_frames]
    feats = spec[:, idxs_max].T
    feats = np.log(feats + 1e-5)

    # get waveshape info
    if 1:
        spec_amps, spec_amps_smooth, peak_idxs_valid = get_waveshape_feats(spec)

        # peak density
        wf_dur = (samp_end - samp_start) / SAMPLERATE
        num_peaks = len(peak_idxs_valid)
        peak_density = num_peaks / wf_dur # peaks per second
        feats = np.hstack((feats, np.ones((feats.shape[0], 1)) * peak_density))



    # vis
    if 0:#tag == 'tags_jingle':
        spec_all = np.abs(stft(wf, WIN_SIZE, HOP_SIZE)) # num_freqs x num_frames

        max_amp = np.max(np.abs(wf))


        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))

        ax0 = fig.add_subplot(4, 1, 1)
        ax0.plot(np.arange(len(wf)) / SAMPLERATE, wf, c='b')
        for s in [samp_start, samp_end]:
            ax0.plot([s / SAMPLERATE] * 2, [-max_amp, max_amp], c='r')

        ax1 = fig.add_subplot(4, 1, 2, sharex=ax0)
        ax1.imshow(spec_all ** 0.4, origin='lower', interpolation='none', aspect='auto',
                   cmap=plt.cm.get_cmap('jet'), extent=[0, len(wf) / SAMPLERATE, 0, SAMPLERATE / 2])

        ax2 = fig.add_subplot(4, 1, 3)
        ax2.imshow(feats, origin='lower', interpolation='none', aspect='auto',
                   cmap=plt.cm.get_cmap('jet'), extent=[0, SAMPLERATE / 2, 0, feats.shape[0] * HOP_SIZE / SAMPLERATE])

        ax3 = fig.add_subplot(4, 1, 4, sharex=ax0)
        ax3.plot(np.arange(len(spec_amps)) * HOP_SIZE / SAMPLERATE + samp_start / SAMPLERATE,
                 spec_amps, c='k', zorder=0)
        ax3.plot(np.arange(len(spec_amps)) * HOP_SIZE / SAMPLERATE + samp_start / SAMPLERATE,
                 spec_amps_smooth, c='r', zorder=1)
        ax3.scatter(np.array(peak_idxs_valid) * HOP_SIZE / SAMPLERATE + samp_start / SAMPLERATE,
                    spec_amps_smooth[peak_idxs_valid], c='g', zorder=2)
        # ax3.set_ylabel(np.round(feats_td_smooth, 2))

        if tag is not None:
            ax0.set_title(tag)

        plt.show()

    return wf_annot, feats

def get_waveshape_feats(spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    spec_amps = np.sum(spec, axis=0)
    len_smooth = 7
    kern_smooth = np.ones(len_smooth) / len_smooth
    spec_amps_smooth = np.convolve(spec_amps, kern_smooth, mode='same')

    num_frames = spec.shape[1]
    peak_halfwin_dur = 0.1
    hop_dur = HOP_SIZE / SAMPLERATE
    min_spec_amp_smooth = 4.0
    peak_halfwin = int(np.round(peak_halfwin_dur / hop_dur))
    peak_crit = (spec_amps_smooth[1:-1] >= spec_amps_smooth[:-2]) * \
                (spec_amps_smooth[1:-1] >= spec_amps_smooth[2:]) * \
                (spec_amps_smooth[1:-1] >= min_spec_amp_smooth)
    peak_idxs = np.where(peak_crit)[0] + 1
    peak_idxs_valid = []
    for i in peak_idxs:
        vals_i = spec_amps_smooth[max(0, i - peak_halfwin):min(num_frames, i + peak_halfwin + 1)]
        if np.all(spec_amps_smooth[i] >= vals_i):
            peak_idxs_valid.append(i)

    return spec_amps, spec_amps_smooth, peak_idxs_valid
