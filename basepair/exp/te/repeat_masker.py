"""
repeat_masker_file = f"{ddir}/raw/annotation/mm10/RepeatMasker/mm10.fa.out.gz"
!mkdir -p {ddir}/raw/annotation/mm10/RepeatMasker/
if not os.path.exists(repeat_masker_file):
    !wget http://www.repeatmasker.org/genomes/mm10/RepeatMasker-rm405-db20140131/mm10.fa.out.gz -O {repeat_masker_file}
"""
import pandas as pd


def wget_repeat_masker(output_file):
    from kipoi.specs import RemoteFile
    RemoteFile("http://www.repeatmasker.org/genomes/mm10/RepeatMasker-rm405-db20140131/mm10.fa.out.gz",
               "c046c8a8d1a1ce20eb865574d31d528b").get_file(output_file)


def read_repeat_masker(file_path):
    """Read the RepeatMasker file
    """
    dfrm = pd.read_table(file_path, delim_whitespace=True, header=[1])

    dfrm.columns = [x.replace("\n", "_") for x in dfrm.columns]

    dfrm['name'] = dfrm['repeat'] + "//" + dfrm['class/family']

    dfrm = dfrm[['ins.', 'sequence', 'begin', 'name']]
    dfrm.columns = ['chrom', 'start', 'end', 'name']
    return dfrm
