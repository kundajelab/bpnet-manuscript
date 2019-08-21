import numpy as np


def kimura_2p_distance(seq1, seq2, verbose=True):
    """
    Kimura 2-Parameter distance = -0.5 log( (1 - 2p -q) * sqrt( 1 - 2q ) )
    where:
    p = transition frequency
    q = transversion frequency

    It's equivalent to model='K80' from R's APE package:
    https://www.rdocumentation.org/packages/ape/versions/5.2/topics/dist.dna

    # Tested in R:
    ```R
        library(magrittr)
        library(ape)
        df = read.table('seq.txt', header=FALSE, stringsAsFactors=FALSE)
        seqs = DNAStringSet(df$V1) %>% as.DNAbin
        dist.dna(seqs, model='K80')  # 2.603
        dist.dna(seqs, model='K81')  # 2.635  -> default
    ```
    """
    from math import log, sqrt
    pairs = []

    for x in zip(seq1, seq2):
        if '-' not in x:
            pairs.append(x)

    ts_count = 0
    tv_count = 0
    length = len(pairs)

    transitions = ["AG", "GA", "CT", "TC"]
    transversions = ["AC", "CA", "AT", "TA",
                     "GC", "CG", "GT", "TG"]

    for (x, y) in pairs:
        if x + y in transitions:
            ts_count += 1
        elif x + y in transversions:
            tv_count += 1

    p = float(ts_count) / length
    q = float(tv_count) / length
    try:
        d = -0.5 * log((1 - 2 * p - q) * sqrt(1 - 2 * q))
    except ValueError:
        if verbose:
            print(f"Tried to take log of a negative number. p={p}, q={q}")
            print(seq1, seq2)
        return np.nan
    return d


def consensus_dist(seqs, seq_distance_fn, **kwargs):
    from concise.utils.pwm import PWM
    from concise.preprocessing.sequence import one_hot2string, DNA
    consensus = PWM(seqs.mean(0) + 0.001).get_consensus()
    return np.array([seq_distance_fn(consensus, s, **kwargs)
                     for s in one_hot2string(seqs, DNA)])


def consensus_dist_kimura(seqs):
    return consensus_dist(seqs, seq_distance_fn=kimura_2p_distance, verbose=False)


def sort_seqs_kimura(seqs):
    return seqs[np.argsort(consensus_dist_kimura(seqs))]
