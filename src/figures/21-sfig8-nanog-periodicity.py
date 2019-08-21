#!/usr/bin/env python3

#Charles McAnanay
#Summer 2019
#Zeitlinger Lab

#Define data paths
nanog_chipnexus_signal_filepath = "data/txt/nanog-periodicity/aggregated-signal.Nanog.txt" #.Txt file with the average ChIP-nexus signal across Nanog motif
nanog_chipnexus_signal_smoothed_filepath = "data/txt/nanog-periodicity/aggregated-signal-smoothed.Nanog.txt"
sasa_filepath = "nucleosomeSasas.dat" #VMD measures of SASA data across a canonical nucleosome as described in the manuscript
AT_frequency_filepath = "AT_freq_across_all_voong_dyads_200bp.txt" #.TXT file that contains the average AA/AT/TT/TA di-nucleotide frequency across nucleosomes defined by Voong et al.

import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams['text.latex.preamble'] = [ r'\usepackage{helvet}', r'\usepackage{sansmath}', r'\sansmath']
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
CELL_1_COLUMN = 85/ 25.4
FIG_WIDTH = CELL_1_COLUMN #inches
FIG_HEIGHT = 2.5 # inches
FONT_SIZE = 8
FONT_NAME='sans-serif'
def getXVals (v, viewSize, offset):
    lhs = (viewSize - v.shape[0] ) // 2 + offset
    xdats = np.arange(lhs, lhs + v.shape[0])
    return xdats

##Nanog periodicity data processing.
#Load in the data from Ziga.
dats = np.loadtxt(nanog_chipnexus_signal_filepath)
#Filter it with a wide gaussian for background subtraction.
#Plot the background subtracted data (and save them for visualization).
subDats = dats
subDats = subDats - min(subDats)
subDats = subDats / max(subDats)
#Note that this isn't normalized from -1 to 1, like the other parameters. This is to avoid the sharp spike in Nanog periodicity dominating everything.
subDats = (subDats - 0.25 ) * 4
subDats = subDats / gaussian_filter(np.abs(subDats), 5)
subDats *= 0.5
np.savetxt("centered.txt", subDats)
dats -= np.loadtxt(nanog_chipnexus_signal_smoothed_filepath)

##Sasa processing.
sasas = np.loadtxt(sasa_filepath)
#trim the ends off of the sasas, since they represent DNA in solution.
#sasas = sasas[10:-10,:]
#Sasas are reported per strand, so combine then since we don't care about strands.
combSasas = sasas[:,0] + sasas[:,1]
#And normalize them to go from 0 to 1.
#How much should the SASA data be offset from the nanog data? An offset of 0 assumes that Nanog binding is dyad-centered.
#I experimented a bit and found that 4 got the two just about centered on each other.
#Since the Nanog data has 200 bp in it, we need to pad out the SASA data to compensate. This is where the offset is used to shift the SASA data around.


##AT frequency processing.
#Now load in the AT frequency data from Melanie.
ATFreqs = np.loadtxt(AT_frequency_filepath)
#I want to use the same offset as for the sasa here - both the sasa and AT data are dyad-centered.

colors = [ (0, 0, 0), (230/255, 159/255, 0), (86/255, 180/255, 233/255)]

fig1 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=600)
offset = 0.05
boffset = 0.13
height = (1-offset*2.5-boffset)/3
loffset = 0.17
width = 0.80
axSasa = fig1.add_axes((loffset, boffset, width, height))
axATFreq = fig1.add_axes((loffset, boffset + offset+height, width, height))
axNanog = fig1.add_axes((loffset, boffset +offset*2+height*2, width, height))

viewSize = dats.shape[0]

def plotVals(ax, xdats, ydats, label, color):
    ax.plot(xdats, gaussian_filter(ydats,2) , '-', label=label, color=color)
    ax.plot(xdats, ydats, 'o-', color=color, label = '(raw)', linewidth=0.2, markersize=0.5)
    ax.grid(axis='x', which='minor')
    ax.tick_params(axis='x', pad=-1, labelsize=FONT_SIZE, bottom=False, top=False, labelbottom= False)
    ax.tick_params(axis='x', which='minor', bottom = False)
    ax.tick_params(axis='y', pad=0, labelsize=FONT_SIZE)
    ax.set_xlim(left = 0, right = 200)
    xticks = [x*10 for x in range(dats.shape[0] // 10 + 1)]
    xMinorTicks = [x*10 for x in range(dats.shape[0] // 10 + 1)]
    ax.set_xticks(xticks, minor = False)
    ax.set_xticks(xMinorTicks, minor = True)
    #ax.set_yticklabels(ax.get_yticks().tolist())
    #for label in ax.get_yticklabels():
    #    label.set_backgroundcolor('red')
    #    label.set_clip_on(False)
    #ax.get_xaxis().set_visible(False)
    #ax.legend(loc='right', fontsize=FONT_SIZE)
    ax.set_ylabel(label, fontsize=FONT_SIZE, labelpad=10)
    ax.yaxis.set_label_coords(-0.10, 0.5)
    ax.set_frame_on(False)
plotVals(axSasa, getXVals(combSasas, viewSize, 0), combSasas, "SASA ($\mathrm{\AA}^2$)", colors[0])
axSasa.set_ylim((50, 150))
plotVals(axATFreq, getXVals(ATFreqs, viewSize, 0), ATFreqs, "AT freq", colors[1])
axATFreq.set_ylim((0.25, 0.35))
plotVals(axNanog, getXVals(dats, viewSize, 0), dats, "Nanog CWM", colors[2])
axNanog.set_ylim((-1, 1))
axSasa.get_xaxis().set_visible(True)
axNanog.text(-0.2,1, "A", fontsize=12, transform=axNanog.transAxes, verticalalignment='top')
axSasa.tick_params(axis='x', pad = 0, labelsize=FONT_SIZE, bottom = False, labelbottom = True)
print(list(axSasa.get_xticklabels()))
print(list(axSasa.get_xticks()))

xtickLabels = [(x-10)*10 if x % 2 == 0 else "" for x in range(dats.shape[0] // 10 + 1)]
#xtickLabels[10] = "dyad"
axSasa.set_xticklabels(xtickLabels, fontname=FONT_NAME)
axSasa.set_xlabel("Position (bp)", fontsize=FONT_SIZE, labelpad=0)
fig1.savefig("sfig8-nanog-periodicity-phasing.png")
plt.close(fig1)

fftNanog = np.fft.fft(dats)
freqsNanog = np.fft.fftfreq(dats.shape[0])
fftSasa = np.fft.fft(combSasas)
freqsSasa = np.fft.fftfreq(combSasas.shape[0])
fftATFreq = np.fft.fft(ATFreqs)
freqsATFreq = np.fft.fftfreq(ATFreqs.shape[0])

fig3 = plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT), dpi=600)
ax3 = fig3.add_axes([0.17, 0.11, 0.80, 0.86])
powerSasa = np.abs(fftSasa)**2
powerATFreq = np.abs(fftATFreq)**2
powerNanog = np.abs(fftNanog)**2
retFreqNanog = 1/freqsNanog
retFreqSasa = 1/freqsSasa
retFreqATFreq = 1/freqsATFreq
def normalizePower(power, periods):
    freqFilter = np.where(np.logical_and(periods < 20, periods > 0))
    return power / max(power[freqFilter])
powerSasa = normalizePower(powerSasa, retFreqSasa)
powerATFreq =  normalizePower(powerATFreq, retFreqATFreq)
powerNanog =  normalizePower(powerNanog, retFreqNanog)
ax3.plot(1/freqsSasa, powerSasa, 'o-', label="SASA", color = colors[0], markersize=3)
ax3.plot(1/freqsATFreq, powerATFreq, 'o-', label="AT freq", color = colors[1], markersize=3)
ax3.plot(1/freqsNanog, powerNanog, 'o-',  label="Nanog", color = colors[2], markersize=3)
ax3.set_xlim(0,20)
ax3.set_ylim(0,1.05)
ax3.legend( fontsize = FONT_SIZE, frameon=False)
ax3.tick_params(axis='both', pad = 0, labelsize=FONT_SIZE, labelbottom = True)
ax3.tick_params(axis='y', labelleft=False, left=False)
ax3.set_xlabel("Period (bp)", fontsize=FONT_SIZE, labelpad=0)
ax3.set_ylabel("Normalized spectral density", fontsize=FONT_SIZE, labelpad=0)
ax3.yaxis.set_label_coords(-0.01, 0.5)
ax3.text(-0.2,1, "B", fontsize=12, transform=ax3.transAxes, verticalalignment='top')
ax3.set_frame_on(False)
fig3.savefig("sfig8-nanog-periodicity-spectral.png")
