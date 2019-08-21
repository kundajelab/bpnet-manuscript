from kipoi.writers import RegionWriter, BedGraphWriter
import numpy as np


def get_chrom_sizes(fasta_file, chromosomes=None):
    """Get chromosome files from a fasta file
    """
    from pysam import FastaFile
    fa = FastaFile(fasta_file)
    if chromosomes is None:
        genome = [(c, l) for c, l in zip(fa.references, fa.lengths)]
    else:
        genome = [(c, l) for c, l in zip(fa.references, fa.lengths)
                  if c in chromosomes]
    return genome


class BigWigWriter(RegionWriter):
    """

    # Arguments
      file_path (str): File path of the output tsv file
      genome_file: genome file containing chromosome sizes. Can
        be None. Can be overriden by `chrom_sizes`.
      chrom_sizes: a list of tuples containing chromosome sizes.
        If not None, it overrided `genome_file`.
      is_sorted: if True, the provided entries need to be sorted beforehand

    Note: One of `genome_file` or `chrom_sizes` shouldn't be None.
    """

    def __init__(self, file_path, genome_file=None, chrom_sizes=None, is_sorted=True):
        import pandas as pd
        import pyBigWig
        self.file_path = file_path
        self.genome_file = genome_file
        # read the genome file
        if chrom_sizes is not None:
            self.chrom_sizes = chrom_sizes
        else:
            if genome_file is None:
                raise ValueError("One of `chrom_sizes` or `genome_file` should not be None")
            self.chrom_sizes = pd.read_csv(self.genome_file, header=None, sep='\t').values.tolist()
        self.bw = pyBigWig.open(self.file_path, "w")
        self.bw.addHeader(self.chrom_sizes)
        self.is_sorted = is_sorted
        if not self.is_sorted:
            import tempfile
            self.bgw = BedGraphWriter(tempfile.mkstemp()[1])
        else:
            self.bgw = None

    def region_write(self, region, data):
        """Write region to file. Note: the written regions need to be sorted beforehand.

        # Arguments
          region: a `kipoi.metadata.GenomicRanges`,  `pybedtools.Interval` or a dictionary with at least keys:
            "chr", "start", "end" and list-values. Example: `{"chr":"chr1", "start":0, "end":4}`.
          data: a 1D-array of values to be written - where the 0th entry is at 0-based "start"
        """
        if not self.is_sorted:
            self.bgw.region_write(region, data)
            return None

        def get_el(obj):
            if isinstance(obj, np.ndarray):
                assert len(data.shape) == 1
            if isinstance(obj, list) or isinstance(obj, np.ndarray):
                assert len(obj) == 1
                return obj[0]
            return obj

        if isinstance(region, dict):
            if 'chr' in region:
                chr = get_el(region["chr"])
            elif 'chrom' in region:
                chr = get_el(region["chr"])
            else:
                raise ValueError("'chr' or 'chrom' not in `region`")
            start = int(get_el(region["start"]))

            if 'end' in region:
                end = int(get_el(region["end"]))
            elif 'stop' in region:
                end = int(get_el(region["end"]))
            else:
                raise ValueError("'end' or 'stop' not in `region`")
        else:
            # works also with pybedtools.Interval
            chr = region['chrom']
            start = region['start']
            end = region['end']

        if end - start != len(data):
            raise ValueError(f"end - start ({end - start})!= len(data) ({len(data)})")
        # if len(data.shape) == 2:
        #     data = data.sum(axis=1)
        assert len(data.shape) == 1

        self.bw.addEntries(chr, int(start), values=data.astype(float), span=1, step=1, validate=True)

    def close(self):
        """Close the file
        """
        if self.is_sorted:
            self.bw.close()
        else:
            # convert bedGraph to bigwig
            from pybedtools import BedTool
            # close the temp file
            self.bgw.close()
            # sort the tempfile and get the path of the sorted file
            sorted_fn = BedTool(self.bgw.file_path).sort().fn
            # write the bigwig file
            with open(sorted_fn) as ifh:
                for l in ifh:
                    chr, start, end, val = l.rstrip().split("\t")
                    self.bw.addEntries([chr], [int(start)], ends=[int(end)], values=[float(val)])
            self.bw.close()
