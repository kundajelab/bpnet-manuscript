import pandas as pd
import numpy as np


def read_chexmix_dfi(expriment_events_path):
    dfi = pd.read_csv(expriment_events_path, sep='\t', skiprows=list(range(6)))
    subtype_cols = dfi.SubtypePoint.str.split(":", expand=True)

    dfi['chrom'] = subtype_cols[0]
    
    dfi['strand'] = subtype_cols[2]
    dfi['pattern_center_abs'] = subtype_cols[1].astype(int)
    # fix the strand offset 
    dfi['pattern_center_abs'] = dfi['pattern_center_abs'] - (dfi['strand'] == '-')
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
    
    # fix the strand offset (strange)
    df['pattern_center_abs'] = interval_center(df).astype(int) - (df['strand'] == '-')
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


# peak size
def read_homer_dfi(fpath, peak_fpath, tf=None, exclude_chr=['chrX', 'chrY'], peak_size=1000):
    from tqdm import tqdm
    import pandas as pd

    df = pd.read_csv(fpath, sep='\t')
    # read the peaks file
    peaks = pd.read_csv(peak_fpath, sep='\t', header=None, names=['chrom', 'start', 'end'])
    
    # TODO - resize start and end
    df = pd.concat([df, peaks.iloc[df.PositionID - 1].reset_index()], axis=1)

    df = df.rename(columns={"MotifScore": 'score', "Strand": 'strand'})
    
    df['pattern_center_abs'] = (df['start'] + df['end']) // 2 + df['Offset']
    
    df['pattern_len'] = df['Sequence'].str.len()
    
    rc_shift = ( df['pattern_len'] - 1) * (df['strand'] == '-')
    df['pattern_start_abs'] = df['pattern_center_abs']  - rc_shift
    df['pattern_end_abs'] = df['pattern_start_abs'] + df['pattern_len']
    
    # re-compute the center position
    add_offset = df.strand.map({"+": 1, "-": 0})
    delta = (df.pattern_end_abs + df.pattern_start_abs) % 2
    df['pattern_center_abs'] = (df.pattern_end_abs + df.pattern_start_abs) // 2 + add_offset * delta
    
    # relative pattern center
    # update start
    center = (df['start'] + df['end']) // 2
    df['start'] = center - peak_size // 2
    df['end'] = center + peak_size // 2 + (peak_size % 2)
 
    df['pattern_center'] = df['pattern_center_abs'] - df['start']
    
    df['example_idx'] = df.PositionID - 1
    
    if tf is not None:
        df['pattern_name'] = tf + '/' + df['Motif Name'].str.split("-", expand=True)[0] 
        
        
    # exclude chromosomes
    keep_idx = pd.Series(np.arange(len(peaks)))[~peaks.chrom.isin(exclude_chr)]
    keep_idx = pd.Series(data=np.arange(len(keep_idx)), index=keep_idx.index)
    example_idx_map = dict(keep_idx)
    
    df = df[df.example_idx.isin(example_idx_map)]
    df['example_idx'] = df['example_idx'].map(example_idx_map)
    return df