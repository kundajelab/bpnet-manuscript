import pandas as pd
import numpy as np


def read_chexmix_dfi(expriment_events_path):
    dfi = pd.read_csv(expriment_events_path, sep='\t', skiprows=list(range(6)))
    subtype_cols = dfi.SubtypePoint.str.split(":", expand=True)

    dfi['chrom'] = subtype_cols[0]
    dfi['pattern_center_abs'] = subtype_cols[1].astype(int)
    dfi['strand'] = subtype_cols[2]
    dfi['profile_score'] = dfi['experiment_log2Fold']
    dfi['motif_score'] = dfi['SubtypeMotifScore']
    dfi['pattern_name'] = (dfi['SubtypeName'].str.replace("Subtype", "").astype(int) + 1).astype(str)
    return dfi


def read_meme_motifs(meme_file):
    from basepair.external.meme import read
    from concise.utils.pwm import PWM

    with open(meme_file) as f:
        record = read(f)
    return {str(i + 1): PWM(pd.DataFrame(m.pwm)[list("ACGT")].values + 0.01,
                            name=m.num_occurrences)
            for i, m in enumerate(record)}


def read_fimo_dfi(file_path):
    from basepair.preproc import interval_center
    from basepair.config import all_chr
    df = pd.read_csv(file_path, sep='\t', comment='#')
    df['chrom'] = df['sequence_name']
    df['end'] = df['stop']
    df['pattern_center_abs'] = interval_center(df).astype(int)
    df['pattern_name'] = df['motif_alt_id'].str.replace("MEME-", "")
    df = df[df.chrom.isin(all_chr)]
    return df



def read_transfac(file_path, ignore_motif_name=True):
    from concise.utils.pwm import PWM
    from collections import defaultdict

    with open(file_path) as f:
        motif_lines = defaultdict(list)
        if not ignore_motif_name:
            motif = None
        else:
            motif = 0
        for line in f:
            if line.startswith("DE"):
                if not ignore_motif_name:
                    motif = line.split("\t")[1].strip()  # all are called motif 1
                else:
                    motif += 1
            elif line.startswith("XX"):
                if not ignore_motif_name:
                    motif = None
            else:
                motif_lines[motif].append(line.split("\t"))

    return {str(motif): PWM(np.array(v)[:, 1:-1].astype(float) + 0.01)
            for motif, v in motif_lines.items()}

def read_homer(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None, sep='\t').values
    l.split(",P:")[1].strip() 
    with open("motif1.motif.txt") as f:
        l = f.readline()
    return PWM(df + 0.01, name="P={}".format(l.split(",P:")[1].strip()))
