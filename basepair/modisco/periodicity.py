"""Functions useful for computing the periodicty
"""
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from basepair.plot.profiles import extract_signal
from basepair.plot.tracks import plot_tracks
from basepair.plot.heatmaps import heatmap_importance_profile, normalize
from typing import Tuple


def periodicity_ft(signal: np.array
                  ) -> Tuple[np.array, np.array]:
    """Compute the periodicity of a genomic signal.
    
    See the following tutorial:
    https://colab.research.google.com/drive/1s158_1RxkKkIIlKrxdCJ3BebH7CvK3pK#scrollTo=kVcWfnr0xSxi

    Args:
      signal: signal at base-resolution.

    Returns 
      A tuple of periodicity (t0) in nucleotide unit and 
        the power spectrum.
    """
    power_spectrum = np.abs(np.fft.rfft(signal))**2
    freq = np.fft.rfftfreq(len(signal))
    # Return everything except the 0-frequency
    # which is equal to:
    # np.abs(np.fft.rfft(signal)[0]) == np.sum(signal)
    
    # Last element in this list is the one with 
    # largest frequency and hence smallest periodicity (t0).
    return 1 / freq[1:], power_spectrum[1:]


def periodicity_ft_summit(t0: np.array,
                          power_spectrum: np.array
                         ) -> Tuple[float, float]:
    """Get the periodicity summit together with the 
    expected precision periodicity +- error.
    
    Args:
      t0, power_spectum: Periodicity and power spectrum 
        returned by periodicity_ft.
      
    Returns:
      Tuple containing periodicity and the +- 10bp error.
    """
    i_max = np.argmax(power_spectrum)
    error = max(t0[i_max] - t0[i_max+1],
                t0[i_max - 1] - t0[i_max]) / 2
    return t0[i_max], error


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def compute_power_spectrum(pattern, task, data):
    seqlets = data.seqlets_per_task[pattern]
    wide_seqlets = [s.resize(data.footprint_width)
                    for s in seqlets
                    if s.center() > data.footprint_width // 2 and
                    s.center() < data.get_seqlen(pattern) - data.footprint_width // 2
                    ]
    p = extract_signal(data.get_region_grad(task, 'profile'), wide_seqlets)

    agg_profile = np.log(np.abs(p).sum(axis=-1).sum(axis=0))

    agg_profile = agg_profile - agg_profile.mean()
    agg_profile = agg_profile / agg_profile.std()

    smooth_part = smooth(agg_profile, 10)
    oscilatory_part = agg_profile - smooth_part

    t0, ps = periodicity_ft(oscilatory_part)
    return ps, t0, 1 / t0


def periodicity_10bp_frac(pattern, task, data):
    ps, t0, freq = compute_power_spectrum(pattern, task, data)
    assert t0[18] == 10.526315789473685
    return ps[18] / ps.sum()


def plot_power_spectrum(pattern, task, data):
    seqlets = data.seqlets_per_task[pattern]
    wide_seqlets = [s.resize(data.footprint_width)
                    for s in seqlets
                    if s.center() > data.footprint_width // 2 and
                    s.center() < data.get_seqlen(pattern) - data.footprint_width // 2
                    ]
    p = extract_signal(data.get_region_grad(task, 'profile'), wide_seqlets)

    agg_profile = np.log(np.abs(p).sum(axis=-1).sum(axis=0))
    heatmap_importance_profile(normalize(np.abs(p).sum(axis=-1)[:500], pmin=50, pmax=99), figsize=(10, 20))
    heatmap_fig = plt.gcf()
    # heatmap_importance_profile(np.abs(p*seq).sum(axis=-1)[:500], figsize=(10, 20))

    agg_profile = agg_profile - agg_profile.mean()
    agg_profile = agg_profile / agg_profile.std()

    smooth_part = smooth(agg_profile, 10)
    oscilatory_part = agg_profile - smooth_part

    avg_fig, axes = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
    axes[0].plot(agg_profile, label='original')
    axes[0].plot(smooth_part, label="smooth", alpha=0.5)
    axes[0].legend()
    axes[0].set_ylabel("Avg. importance")
    axes[0].set_title("Average importance score")
    # axes[0].set_xlabel("Position");
    axes[1].plot(oscilatory_part)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("original - smooth")
    avg_fig.subplots_adjust(hspace=0)  # no space between plots
    # plt.savefig('nanog-agg-profile.png', dpi=300)
    # plt.savefig('nanog-agg-profile.pdf')

    fft_fig = plt.figure(figsize=(11, 2))
    t0, power_spectrum = periodicity_ft(oscilatory_part)
    plt.plot(t0, power_spectrum, "-o")
    plt.xlim([0, 50])
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(25, integer=True))
    plt.grid(alpha=0.3)
    plt.xlabel("1/frequency (bp)")
    plt.ylabel("Power spectrum")
    plt.title("Power spectrum")
    plt.gcf().subplots_adjust(bottom=0.4)
    return heatmap_fig, avg_fig, fft_fig