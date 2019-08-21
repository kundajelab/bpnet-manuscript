from pybedtools import BedTool
import pandas as pd
from kipoi.specs import RemoteFile
from pathlib import Path


def download_repeat_masker(repeat_masker_file=None):
    if repeat_masker_file is None:
        from basepair.config import get_data_dir
        ddir = get_data_dir()
        repeat_masker_file = Path(f"{ddir}/raw/annotation/mm10/RepeatMasker/mm10.fa.out.gz")
    repeat_masker_file.parent.mkdir(exist_ok=True)
    return RemoteFile('http://www.repeatmasker.org/genomes/mm10/RepeatMasker-rm405-db20140131/mm10.fa.out.gz',
                      md5='c046c8a8d1a1ce20eb865574d31d528b').get_file(repeat_masker_file)


def read_repeat_masker(file_path):
    dfrm = pd.read_table(file_path, delim_whitespace=True, header=[1])

    dfrm.columns = [x.replace("\n", "_") for x in dfrm.columns]

    dfrm['name'] = dfrm['repeat'] + "//" + dfrm['class/family']

    dfrm = dfrm[['ins.', 'sequence', 'begin', 'name']]
    dfrm.columns = ['chrom', 'start', 'end', 'name']
    return dfrm


def intersect_repeat_masker(pattern_name, seqlets: BedTool, repeat_masker: BedTool, f=1.0):
    """Intersect the seqlets bed file with
    """
    try:
        dfint = seqlets.intersect(repeat_masker, wa=True, wb=True, f=f).to_dataframe()
    except Exception:
        return None
    t = dfint.blockCount.str.split("//", expand=True)
    dfint['pattern_name'] = pattern_name
    dfint['repeat_name'] = t[0]
    dfint['repeat_family'] = t[1]
    dfint['n_pattern'] = seqlets.to_dataframe()[['chrom', 'start', 'end']].drop_duplicates().shape[0]
    dfint['interval'] = dfint['chrom'] + ":" + dfint['start'].astype(str) + "-" + dfint['end'].astype(str)
    return dfint[['chrom', 'start', 'end', 'interval', 'pattern_name', 'n_pattern', 'repeat_name', 'repeat_family']]
