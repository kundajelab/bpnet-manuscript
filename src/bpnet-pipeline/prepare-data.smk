"""Prepare all the required data for modeling

Move it to Oak
"""

from pathlib import Path
tfs = ['Oct4', 'Sox2', 'Nanog', 'Klf4']
chipseq_tfs = ['Oct4', 'Sox2', 'Nanog']
strands = ['pos', 'neg']


def strand2long(s):
    if s == 'pos':
        return 'positive'
    elif s == 'neg':
        return 'negative'
    else:
        raise ValueError("wrong s")


ddir = Path('/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/data/')
sdir = Path('/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/software/')

NEXUS_PIPELINE_DIR = '/srv/scratch/amr1/chip-nexus-pipeline/cromwell-executions/chip_nexus'
CHIPSEQ_PIPELINE_DIR = '/srv/scratch/amr1/chip-seq-pipeline2/cromwell-executions/chip/'

GENOME_FILE = ddir / 'mm10.chrom.sizes'
FASTA_FILE = ddir / 'mm10_no_alt_analysis_set_ENCODE.fasta'


chipnexus_dir = ddir / 'chip-nexus/'

chipseq_dir = ddir / 'chip-seq/'

rule all:
    input:
        # ChIP-nexus
        expand(os.path.join(chipnexus_dir, '{tf}/counts.{strand}.bw'),
               tf=tfs + ['patchcap'],
               strand=strands),
        expand(os.path.join(chipnexus_dir, '{tf}/idr-optimal-set.summit.bed.gz'),
               tf=tfs,
               strand=strands),
        # ChIP-seq
        expand(os.path.join(chipseq_dir, '{tf}/counts.{strand}.bw'),
               tf=chipseq_tfs + ['input-control'],
               strand=strands),
        expand(os.path.join(chipseq_dir, '{tf}/idr-optimal-set.summit.bed.gz'),
               tf=chipseq_tfs,
               strand=strands),
        expand(os.path.join(ddir, 'chip-{assay}/{tfs}.50-percent-overlap.tsv.gz'),
               zip,
               assay=['seq', 'nexus', 'nexus'],
               tfs=['OSN', 'OSN', 'OSNK'],
               )

# --------------------------------------------
# ChIP-nexus
rule chip_nexus_copy_bigwig:
    """Copy the bigwig files
    """
    input:
        bw = lambda wildcards: os.path.join(NEXUS_PIPELINE_DIR,
                                            wildcards.tf.lower(),
                                            "call-count_signal_track_pooled/execution",
                                            f"mesc_{wildcards.tf.lower()}_nexus_1.trim."
                                            f"merged.nodup.pooled.{strand2long(wildcards.strand)}.bigwig"
                                            )
    output:
        bw = os.path.join(chipnexus_dir, '{tf}/counts.{strand}.bw')
    shell:
        "cp {input.bw} {output.bw}"


rule chip_nexus_copy_peak:
    """Copy the IDR optimal peak
    """
    input:
        bed = lambda wildcards: os.path.join(NEXUS_PIPELINE_DIR,
                                             wildcards.tf.lower(),
                                             "call-reproducibility_idr/execution/optimal_peak.narrowPeak.gz")
    output:
        bed = os.path.join(chipnexus_dir, '{tf}/idr-optimal-set.narrowPeak.gz')
    shell:
        "cp {input.bed} {output.bed}"


rule chip_nexus_peak_summit:
    """Extract peak summit
    """
    input:
        bed = os.path.join(chipnexus_dir, '{tf}/idr-optimal-set.narrowPeak.gz')
    output:
        bed = os.path.join(chipnexus_dir, '{tf}/idr-optimal-set.summit.bed.gz')
    shell:
        """zcat {input.bed} | awk '{{ print $1 "\t" $2+$10 "\t" $2+$10+1; }}' | gzip -nc > {output.bed}"""


rule generate_nexus_bedGraph:
    """Generate the bigwig files
    """
    input:
        genome_file = GENOME_FILE,
        ta = os.path.join(NEXUS_PIPELINE_DIR,
                          # Use patchcap
                          'patchcap',
                          "call-bam2ta/shard-0/execution/",
                          "mesc_patchcap_nexus_1.trim.merged.nodup.tagAlign.gz")
    params:
        plus_minus = lambda wildcards: {"pos": "+", "neg": "-"}[wildcards.strand]
    output:
        bedGraph = os.path.join(chipnexus_dir, 'patchcap/counts.{strand}.bedGraph')
    shell:
        """
        export LANG=C
        export LC_COLLATE=C
        zcat {input.ta} | sort -k1,1 -k2,2n | \
          bedtools genomecov -5 -bg -strand {params.plus_minus} -g {input.genome_file} -i stdin | \
          sort -k1,1 -k2,2n > {output.bedGraph}
        """

rule create_nexus_bigwig:
    """ sorted.bedGraph -> bw
    """
    input:
        genome_file = GENOME_FILE,
        bedGraph = os.path.join(chipnexus_dir, 'patchcap/counts.{strand}.bedGraph')
    output:
        bw = os.path.join(chipnexus_dir, 'patchcap/counts.{strand}.bw')
    shell:
        """
        bedGraphToBigWig {input.bedGraph} {input.genome_file} {output.bw}
        """

# --------------------------------------------
# ChIP-seq

rule chip_seq_copy_peak:
    """Copy the IDR optimal peak
    """
    input:
        bed = lambda wildcards: os.path.join(CHIPSEQ_PIPELINE_DIR,
                                             wildcards.tf.lower(),
                                             "call-reproducibility_idr/execution/optimal_peak.gz")
    output:
        bed = os.path.join(chipseq_dir, '{tf}/idr-optimal-set.narrowPeak.gz')
    shell:
        "cp {input.bed} {output.bed}"


rule chip_seq_peak_summit:
    """Extract peak summit
    """
    input:
        bed = os.path.join(chipseq_dir, '{tf}/idr-optimal-set.narrowPeak.gz')
    output:
        bed = os.path.join(chipseq_dir, '{tf}/idr-optimal-set.summit.bed.gz')
    shell:
        """zcat {input.bed} | awk '{{ print $1 "\t" $2+$10 "\t" $2+$10+1; }}' | gzip -nc > {output.bed}"""


def get_tagalign(wildcards):
    if wildcards.tf == 'input-control':
        return os.path.join(CHIPSEQ_PIPELINE_DIR,
                            "nanog/call-pool_ta_ctl/execution/mesc_wce_chipseq_rep1.merged.nodup.pooled.tagAlign.gz")
    else:
        tfl = wildcards.tf.lower()
        repname = "" if wildcards.tf == 'Sox2' else 'rep'

        return os.path.join(CHIPSEQ_PIPELINE_DIR, tfl,
                            f"call-pool_ta/execution/mesc_{tfl}"
                            f"_chipseq_{repname}1.merged.nodup.pooled.tagAlign.gz")


rule generate_bedGraph:
    """Generate the bigwig files
    """
    input:
        genome_file = GENOME_FILE,
        ta = get_tagalign
    params:
        plus_minus = lambda wildcards: {"pos": "+", "neg": "-"}[wildcards.strand]
    output:
        bedGraph = os.path.join(chipseq_dir, '{tf}/counts.{strand}.bedGraph')
    shell:
        """
        export LANG=C
        export LC_COLLATE=C
        zcat {input.ta} | sort -k1,1 -k2,2n | \
          bedtools genomecov -5 -bg -strand {params.plus_minus} -g {input.genome_file} -i stdin | \
          sort -k1,1 -k2,2n > {output.bedGraph}
        """

rule create_bigwig:
    """ sorted.bedGraph -> bw
    """
    input:
        genome_file = GENOME_FILE,
        bedGraph = os.path.join(chipseq_dir, '{tf}/counts.{strand}.bedGraph')
    output:
        bw = os.path.join(chipseq_dir, '{tf}/counts.{strand}.bw')
    shell:
        """
        bedGraphToBigWig {input.bedGraph} {input.genome_file} {output.bw}
        """


# --------------------------------------------
# Label the regions

rule label_regions:
    """Generate the genome-wide labels
    """
    input:
        genome_file = GENOME_FILE,
        tasks = 'ChIP-{assay}-tasks.{tfs}.tsv',
    output:
        tsv = os.path.join(ddir, 'chip-{assay}/{tfs}.50-percent-overlap.tsv.gz')
    threads: 10
    shell:
        """
        genomewide_labels \
            --task_list {input.tasks} \
            --outf {output.tsv} \
            --output_type gzip \
            --chrom_sizes {input.genome_file} \
            --bin_stride 50 \
            --left_flank 400 \
            --right_flank 400 \
            --threads {threads} \
            --subthreads 4 \
            --allow_ambiguous \
            --overlap_thresh 0.5 \
            --labeling_approach peak_percent_overlap_with_bin_classification
        """
