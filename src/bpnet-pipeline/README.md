### Setup

In case you are not on the Sherlock cluster, fix the file paths by searching for `/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/data` in all the text files in this directory and replace them with your own paths to the raw data.

### Fetch the data from the ChIP-nexus/seq pipeline output (unnecessary)

`snakemake -s prepare-data.smk`

This command is unnecessary as the data you downloaded by following the steps in [../../README.md](../../README.md) are already the output of `prepare-data.smk`.

### Train BPNet models, generate importance scores, run TF-MoDISco, get motif instances with CWM scanning

Run `snakemake`.

This will train BPNet models and run TF-MoDISco for models specified in the following spreadsheet: <https://docs.google.com/spreadsheets/d/1obNkUbUJZqAJZnGuK8Tw4LyY15srUQ54Ssbg2XV5Za0>. High number of models is due to the hyper-parameter and model comparisons.

#### Running `snakemake` with SLURM

If you would like to run the snakemake rules through the SLURM scheduler, perform the following steps

1. `git clone git@github.com:Avsecz/snakemake-profiles.git ~/.config/snakemake`
  - This will get the snakemake SLURM profile from https://github.com/Avsecz/snakemake-profiles.
2. Run `snakemake --profile slurm --jobs 20 -p`
  - This will submit 20 slurm jobs simultaneusly. Resources for each job (CPU, MEM, GPU, and runtime) are dynamically specified according to the [Snakefile](Snakefile).
