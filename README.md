# BPNet manuscript

Code accompanying the BPNet manuscript.

If you want to use BPNet on your own data, please use the BPNet python package: <https://github.com/kundajelab/bpnet>.

## Folder organization

- `basepair` - python package (contains python functions/classes common across multiple notebooks)
- `src` - scripts for running the experiments and producing the figures
  - `bpnet-pipeline` - Train BPNet models, generate importance scores, run TF-MoDISco, get motif instances with CWM scanning
  - `motif-interactions` - Generate the *in silico* motif interactions.
  - `comparison` - Run ChExMix
  - `figures` - Generate all the paper figures
- `data` - data files
- `tests` - Unit tests.

## Reproducing the results

### 1. Setup the environment

1. Install miniconda or anaconda. 
1. Install git-lfs: `conda install -c conda-forge git-lfs && git lfs install`
1. Clone this repository: `git clone https://github.com/kundajelab/bpnet-manuscript.git && cd bpnet-manuscript`
1. Run: `git lfs pull '-I data/**'` 
1. Run: `conda env create -f conda-env.yaml` (if you want to use the GPU, rename `tensorflow` to `tensorflow-gpu` and make sure you have the correct CUDA version installed to run tensorflow 1.7 or 1.6). This will install a new conda environment `bpnet-manuscript`
1. Activate the environment: `source activate bpnet-manuscript`
1. Install the `basepair` python package for this repository : `pip install -e .`

To speed-up data-loading build [vmtouch](https://hoytech.com/vmtouch/). This is used to
load the bigWig files into system memory cache which allows multiple processes to access
the bigWigs loaded into memory.

Here's how I install vmtouch:

```bash
# ~/bin = directory for localy compiled binaries
mkdir -p ~/bin  
cd ~/bin
# Clone and build
git clone https://github.com/hoytech/vmtouch.git vmtouch_src
cd vmtouch_src
make
# Move the binary to ~/bin
cp vmtouch ../
# Add ~/bin to $PATH
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
```

To make sure saving the Keras model in HDF5 file format works (https://github.com/h5py/h5py/issues/1082), add the following to your `~/.bashrc`:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

### 2. Download the data

- [output.tar.gz](https://zenodo.org/record/3371164)
- [data.tar.gz](https://zenodo.org/record/3371216)

First, make a directory on your machine:

```bash
mkdir -p bpnet-manuscript-data
cd bpnet-manuscript-data
1```

All the data will be downloaded to this directory. In the code-base, replace `/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison` path with your the absolute path of `bpnet-manuscript-data` directory.


#### Download raw data

```bash
wget 'https://zenodo.org/record/3371164/files/output.tar.gz?download=1' -O output.tar.gz && tar xvfz output.tar.gz && rm output.tar.gz
```

#### Download outputs

```bash
wget 'https://zenodo.org/record/3371216/files/data.tar.gz?download=1' -O data.tar.gz && tar xvfz data.tar.gz && rm data.tar.gz
```

### 3. Run all scripts for which the main data were not provided

1. Compute the contribution score files (`output/*/deeplift.imp_score.h5`) as follows:
```bash
source activate bpnet-manuscript
for out in $(ls -d output/*/)
do 
  basepair imp-score-seqmodel ${out%%/} ${out%%/}/deeplift.imp_score.h5 \
          --dataspec ${out%%/}/dataspec.yaml \
          --gpu 0 \
          --batch-size 16 \
          --method deeplift \
          --intp-pattern '*' \
          --peak-width 1000} \
          --seq-width 1000 \
          --memfrac 1 \
          --num-workers 5 \
          --exclude-chr chrX,chrY
done
```
2. Run chexmix
    - Follow the instructions in [src/comparison/README.md](src/comparison/README.md).
	
	
### 4. (Optional) Re-run the remaining computationally heavy scripts 

These steps are optional as the output data were already downloaded in the previous step.

1. Train BPNet, compute contrib. scores, run TF-MoDISco, get motif instances
    - Follow the instructions in [src/bpnet-pipeline/README.md](src/bpnet-pipeline/README.md).
2.  Filter motif instances
    - Follow the instructions in [src/figures/README.md](src/figures/README.md).
3. Motif interaction analysis
    - Follow the instructions in [src/motif-interactions/README.md](src/motif-interactions/README.md).

### 5. Generate figures

Execute notebooks in folder [src/figures/README.md](src/figures/README.md). Figures will be generated to `data/figures`.
