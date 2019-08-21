"""Module containing code for perturbation analysis
"""
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from kipoi.data import Dataset
import matplotlib.pyplot as plt
from basepair.utils import remove_exists
from basepair.stats import symmetric_kl
from basepair.modisco.results import Seqlet
from basepair.data import NumpyDataset
from scipy.stats import wilcoxon


def get_reference_profile(mr, pattern, profiles, tasks, profile_width=70, trim_frac=0.08, seqlen=1000):
    """Generate the reference profile
    """
    from basepair.modisco.results import resize_seqlets
    from basepair.plot.profiles import extract_signal
    seqlets_ref = mr._get_seqlets(pattern, trim_frac=trim_frac)
    seqlets_ref = resize_seqlets(seqlets_ref, profile_width, seqlen=seqlen)

    out = {}
    for task in tasks:
        seqlet_profile_ref = extract_signal(profiles[task], seqlets_ref)
        avg_profile = seqlet_profile_ref.mean(axis=0)
        out[task] = avg_profile
        # metrics_ref = pd.DataFrame([profile_sim_metrics(avg_profile, cp) for cp in seqlet_profile_ref])
    return out


def random_seq_onehot(l):
    """Generate random sequence one-hot-encoded

    Args:
      l: sequence length
    """
    from concise.preprocessing import encodeDNA
    return encodeDNA([''.join(random.choices("ACGT", k=int(l)))])[0]


class PerturbSeqDataset(Dataset):

    def __init__(self, dfi, seqs):
        self.dfi = dfi
        self.seqs = seqs

    def __len__(self):
        return len(self.dfi)

    def __getitem__(self, idx):
        inst = self.dfi.iloc[idx]
        assert inst.row_idx == idx
        ref_seq = self.seqs[inst.example_idx]
        # generate the alternative sequence
        alt_seq = ref_seq.copy()
        alt_seq[int(inst.pattern_start):int(inst.pattern_end)] = random_seq_onehot(inst.pattern_end - inst.pattern_start)

        return alt_seq


class DoublePerturbSeqDataset(Dataset):

    def __init__(self, dfab, seqs):
        """Pertub pairs of motifs

        Args:
          dfab: motif pair data-frame
          seqs: original sequences
        """
        self.dfab = dfab
        self.seqs = seqs

    def __len__(self):
        return len(self.dfab)

    def __getitem__(self, idx):
        inst = self.dfab.iloc[idx]
        ref_seq = self.seqs[inst.example_idx]
        # generate the alternative sequence
        alt_seq = ref_seq.copy()
        alt_seq[int(inst.pattern_start_x):int(inst.pattern_end_x)] = random_seq_onehot(inst.pattern_end_x - inst.pattern_start_x)
        alt_seq[int(inst.pattern_start_y):int(inst.pattern_end_y)] = random_seq_onehot(inst.pattern_end_y - inst.pattern_start_y)
        return alt_seq


# Extracting profiles

class ParturbationDataset(Dataset):

    def __init__(self,
                 dfab: pd.DataFrame,
                 ref: NumpyDataset,
                 single_mut: NumpyDataset,
                 double_mut: NumpyDataset,
                 profile_width=70):
        """
        Args:
          dfab: output constructed by `motif_pair_dfi`
          ref: NumpyDataset
          single_mut: NumpyDataset
        """
        self.dfab = dfab
        self.ref = ref
        self.single_mut = single_mut
        self.double_mut = double_mut
        self.profile_width = profile_width

        # check that the widths are all the same
        if not len(np.unique(self.dfab.pattern_end_x - self.dfab.pattern_start_x)) == 1:
            raise ValueError("All motifs need to have the same width")

        if not len(np.unique(self.dfab.pattern_end_y - self.dfab.pattern_start_y)) == 1:
            raise ValueError("All motifs y need to have the same width")

    def __len__(self):
        return len(self.dfab)

    def extract(self, narrow_seqlet, this_row_idx, other_row_idx, motif_pair_idx):
        def extract_fn(data, seqlet):
            ret = data[int(seqlet.seqname)][int(seqlet.start):int(seqlet.end)]
            if seqlet.strand == '-':
                if ret.ndim == 1:
                    return ret[::-1]
                elif ret.ndim == 2:
                    return ret[::-1, ::-1]
                else:
                    raise ValueError("Don't know how to handle ndim > 2")
            else:
                return ret

        def all_fn(data, seqlet):
            ret = data[int(seqlet.seqname)]
            return ret

        wide_seqlet = narrow_seqlet.resize(self.profile_width)
        return {
            "ref": {
                "narrow": self.ref.dapply(extract_fn, seqlet=narrow_seqlet),
                "wide": self.ref.dapply(extract_fn, seqlet=wide_seqlet),
                "whole": self.ref.dapply(all_fn, seqlet=wide_seqlet)
            },
            "dthis": {
                "narrow": self.single_mut.dapply(extract_fn, seqlet=narrow_seqlet.set_seqname(this_row_idx)),
                "wide": self.single_mut.dapply(extract_fn, seqlet=wide_seqlet.set_seqname(this_row_idx)),
                "whole": self.single_mut.dapply(all_fn, seqlet=wide_seqlet.set_seqname(this_row_idx))
            },
            "dother": {
                "narrow": self.single_mut.dapply(extract_fn, seqlet=narrow_seqlet.set_seqname(other_row_idx)),
                "wide": self.single_mut.dapply(extract_fn, seqlet=wide_seqlet.set_seqname(other_row_idx)),
                "whole": self.single_mut.dapply(all_fn, seqlet=wide_seqlet.set_seqname(other_row_idx))
            },
            "dboth": {
                "narrow": self.double_mut.dapply(extract_fn, seqlet=narrow_seqlet.set_seqname(motif_pair_idx)),
                "wide": self.double_mut.dapply(extract_fn, seqlet=wide_seqlet.set_seqname(motif_pair_idx)),
                "whole": self.double_mut.dapply(all_fn, seqlet=wide_seqlet.set_seqname(motif_pair_idx))
            }}

    def __getitem__(self, idx):
        pair = self.dfab.iloc[idx]
        narrow_seqlet_x = Seqlet(seqname=pair.example_idx,
                                 start=pair.pattern_start_x,
                                 end=pair.pattern_end_x,
                                 name="",
                                 strand=pair.strand_x)
        narrow_seqlet_y = Seqlet(seqname=pair.example_idx,
                                 start=pair.pattern_start_y,
                                 end=pair.pattern_end_y,
                                 name="",
                                 strand=pair.strand_y)
        return {
            "x": self.extract(narrow_seqlet_x,
                              this_row_idx=pair.row_idx_x,
                              other_row_idx=pair.row_idx_y,
                              motif_pair_idx=pair.motif_pair_idx),
            "y": self.extract(narrow_seqlet_y,
                              this_row_idx=pair.row_idx_y,
                              other_row_idx=pair.row_idx_x,
                              motif_pair_idx=pair.motif_pair_idx)
        }


# --------------------------------------------
# old-type-perturbation

class PerturbDataset(Dataset):

    def __init__(self, dfi, seqs, preds, profiles, ref_imp_scores_contrib,
                 alt_dataset, alt_seqs, alt_preds, alt_imp_scores_contrib,
                 ref_profiles,
                 profile_mapping):
        self.dfi = dfi
        self.seqs = seqs
        self.preds = preds
        self.profiles = profiles
        self.ref_imp_scores_contrib = ref_imp_scores_contrib
        self.alt_dataset = alt_dataset
        self.alt_seqs = alt_seqs
        self.alt_preds = alt_preds
        self.alt_imp_scores_contrib = alt_imp_scores_contrib
        self.ref_profiles = ref_profiles
        self.profile_mapping = profile_mapping

    def __len__(self):
        return len(self.dfi)

    def get_change(self, mutated_seqlet_idx, signal_seqlet_idx):
        inst = self.dfi.iloc[signal_seqlet_idx]

        assert inst.row_idx == signal_seqlet_idx

        task = self.profile_mapping[inst.pattern_name]

        ref_profile = self.ref_profiles[inst.pattern_name][task]

        mutated_inst = self.dfi.iloc[mutated_seqlet_idx]
        assert inst.example_idx == mutated_inst.example_idx

        narrow_seqlet = Seqlet(inst.example_idx, inst.pattern_start, inst.pattern_end,
                               name=inst.pattern, strand=inst.strand)
        wide_seqlet = narrow_seqlet.resize(70)

        # ref
        ref_preds = self.preds[task][inst.example_idx]  # all predictions
        ref_preds_seqlet = wide_seqlet.extract(self.preds[task])
        ref_preds_inside = ref_preds_seqlet.sum()
        ref_preds_outside = ref_preds.sum() - ref_preds_inside

        ref_obs = self.profiles[task][inst.example_idx]  # all predictions
        ref_obs_seqlet = wide_seqlet.extract(self.profiles[task])
        ref_obs_inside = ref_obs_seqlet.sum()
        ref_obs_outside = ref_obs.sum() - ref_obs_inside
        try:
            ref_preds_match = symmetric_kl(ref_preds_seqlet, ref_profile).mean()  # compare with the reference
        except Exception:
            ref_preds_match = np.nan

        # ref imp
        ref_imp_scores = self.ref_imp_scores_contrib[f"{task}/weighted"][inst.example_idx]
        ref_imp_scores_seqlet = narrow_seqlet.extract(self.ref_imp_scores_contrib[f"{task}/weighted"])
        ref_imp_inside = ref_imp_scores_seqlet.sum()  # sum in the seqlet region
        ref_imp_outside = ref_imp_scores.sum() - ref_imp_inside  # total - seqlet

        # ref imp counts
        ref_imp_scores_c = self.ref_imp_scores_contrib[f"{task}/count"][inst.example_idx]
        ref_imp_scores_seqlet_c = narrow_seqlet.extract(self.ref_imp_scores_contrib[f"{task}/count"])
        ref_imp_inside_c = ref_imp_scores_seqlet_c.sum()  # sum in the seqlet region
        ref_imp_outside_c = ref_imp_scores_c.sum() - ref_imp_inside_c  # total - seqlet

        # alt
        narrow_seqlet.seqname = mutated_seqlet_idx  # change the sequence name
        wide_seqlet.seqname = mutated_seqlet_idx

        alt_preds = self.alt_preds[task][mutated_seqlet_idx]
        alt_preds_seqlet = wide_seqlet.extract(self.alt_preds[task])
        alt_preds_inside = alt_preds_seqlet.sum()  # sum in the seqlet region
        alt_preds_outside = alt_preds.sum() - alt_preds_inside  # total - seqlet
        try:
            alt_preds_match = symmetric_kl(alt_preds_seqlet, ref_profile).mean()
        except Exception:
            alt_preds_match = np.nan

        alt_imp_scores = self.alt_imp_scores_contrib[f"{task}/weighted"][mutated_seqlet_idx]
        alt_imp_scores_seqlet = narrow_seqlet.extract(self.alt_imp_scores_contrib[f"{task}/weighted"])
        alt_imp_inside = alt_imp_scores_seqlet.sum()  # sum in the seqlet region
        alt_imp_outside = alt_imp_scores.sum() - alt_imp_inside  # total - seqlet

        alt_imp_scores_c = self.alt_imp_scores_contrib[f"{task}/count"][mutated_seqlet_idx]
        alt_imp_scores_seqlet_c = narrow_seqlet.extract(self.alt_imp_scores_contrib[f"{task}/count"])
        alt_imp_inside_c = alt_imp_scores_seqlet_c.sum()  # sum in the seqlet region
        alt_imp_outside_c = alt_imp_scores_c.sum() - alt_imp_inside_c  # total - seqlet

        return {
            "ref": {
                "obs": {
                    "inside": ref_obs_inside,
                    "outside": ref_obs_outside
                },
                "pred": {
                    "inside": ref_preds_inside,
                    "outside": ref_preds_outside,
                    "match": ref_preds_match
                },
                "imp": {
                    "inside": ref_imp_inside,
                    "outside": ref_imp_outside,
                },
                "impcount": {
                    "inside": ref_imp_inside_c,
                    "outside": ref_imp_outside_c,
                },
            },
            "alt": {
                "pred": {
                    "inside": alt_preds_inside,
                    "outside": alt_preds_outside,
                    "match": alt_preds_match
                },
                "imp": {
                    "inside": alt_imp_inside,
                    "outside": alt_imp_outside,
                },
                "impcount": {
                    "inside": alt_imp_inside_c,
                    "outside": alt_imp_outside_c,
                },
            },
        }


class SingleMotifPerturbDataset(Dataset):

    def __init__(self, pdata):
        self.pdata = pdata

    def __len__(self):
        return len(self.pdata.dfi)

    def __getitem__(self, idx):
        return self.pdata.get_change(idx, idx)


class OtherMotifPerturbDataset(Dataset):

    def __init__(self, pdata, dfab):
        self.pdata = pdata
        self.dfab = dfab

    def __len__(self):
        return len(self.dfab)

    def __getitem__(self, idx):
        xidx = int(self.dfab.iloc[idx].row_idx_x)
        yidx = int(self.dfab.iloc[idx].row_idx_y)
        return {"xy": self.pdata.get_change(xidx, yidx),  # mutate x, measure y
                "yx": self.pdata.get_change(yidx, xidx)}  # mutate y, measure x


class DoubleMotifPerturbDataset(Dataset):

    def __init__(self, dfab, dalt_preds, ref_profiles, profile_mapping):
        """Use the predictions from DoublePerturbDatasetSeq

        Args:
          dfab: motif pair dataset
          dalt_preds: predictions from sequences obtained from DoublePerturbDatasetSeq
          ref_profiles: reference profiles for background seqlets
        """
        self.dfab = dfab
        self.ref_profiles = ref_profiles
        self.dalt_preds = dalt_preds
        self.profile_mapping = profile_mapping

    def __len__(self):
        return len(self.dfab)

    def extract_features(self, idx, pattern_name, pattern_start, pattern_end, pattern, strand):
        task = self.profile_mapping[pattern_name]
        ref_profile = self.ref_profiles[pattern_name][task]
        narrow_seqlet = Seqlet(idx, pattern_start, pattern_end,
                               name=pattern, strand=strand)
        wide_seqlet = narrow_seqlet.resize(70)
        ref_preds = self.dalt_preds[task][idx]  # all predictions
        ref_preds_seqlet = wide_seqlet.extract(self.dalt_preds[task])
        ref_preds_inside = ref_preds_seqlet.sum()
        ref_preds_total = ref_preds.sum()
        ref_preds_outside = ref_preds_total - ref_preds_inside
        try:
            ref_preds_match = symmetric_kl(ref_preds_seqlet, ref_profile).mean()  # compare with the reference
        except Exception:
            ref_preds_match = np.nan
        return {
            "pred": {
                "inside": ref_preds_inside,
                "outside": ref_preds_outside,
                "total": ref_preds_total,
                "match": ref_preds_match
            }}

    def __getitem__(self, idx):
        inst = self.dfab.iloc[idx]
        return {"dxy": {
            "x": self.extract_features(idx,
                                       pattern_name=inst.pattern_name_x,
                                       pattern_start=int(inst.pattern_start_x),
                                       pattern_end=int(inst.pattern_end_x),
                                       pattern=inst.pattern_x,
                                       strand=inst.strand_x),
            "y": self.extract_features(idx,
                                       pattern_name=inst.pattern_name_y,
                                       pattern_start=int(inst.pattern_start_y),
                                       pattern_end=int(inst.pattern_end_y),
                                       pattern=inst.pattern_y,
                                       strand=inst.strand_y)
        }}


def generate_data(bpnet, dfi, seqs, profiles, pairs, tasks, output_dir):
    from basepair.exp.chipnexus.spacing import motif_pair_dfi
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # get predictions
    preds = bpnet.predict(seqs)

    # get importance scores
    ref_imp_scores = bpnet.imp_score_all(seqs, method='deeplift', aggregate_strand=True)
    ref_imp_scores_contrib = {k: v * seqs for k, v in ref_imp_scores.items()}

    ref = NumpyDataset({t: {
        "obs": profiles[t],
        "pred": preds[t],
        "imp": {
            "profile": ref_imp_scores_contrib[f"{t}/weighted"],
            "count": ref_imp_scores_contrib[f"{t}/count"],
        }} for t in tasks}, attrs={'index': 'example_idx'})

    # get the interesting motif location
    dfi_subset = (dfi.query('match_weighted_p > 0.2')
                     .query('imp_weighted_p > 0'))
    dfi_subset['row_idx'] = np.arange(len(dfi_subset)).astype(int)

    # single-mutation
    alt_seqs = PerturbSeqDataset(dfi_subset, seqs).load_all()
    alt_preds = bpnet.predict(alt_seqs)
    alt_imp_scores = bpnet.imp_score_all(alt_seqs, method='deeplift', aggregate_strand=True, batch_size=256)
    alt_imp_scores_contrib = {k: v * alt_seqs for k, v in alt_imp_scores.items()}

    single_mut = NumpyDataset({t: {
        "pred": alt_preds[t],
        "imp": {
            "profile": alt_imp_scores_contrib[f"{t}/weighted"],
            "count": alt_imp_scores_contrib[f"{t}/count"],
        }} for t in tasks}, attrs={'index': 'row_idx'})

    # create motif pairs
    dfab = pd.concat([motif_pair_dfi(dfi_subset, motif_pair).assign(motif_pair='<>'.join(motif_pair))
                      for motif_pair in pairs], axis=0)
    dfab['motif_pair_idx'] = np.arange(len(dfab))

    dpdata_seqs = DoublePerturbSeqDataset(dfab, seqs).load_all(num_workers=10)
    double_alt_preds = bpnet.predict(dpdata_seqs)
    double_alt_imp_scores = bpnet.imp_score_all(dpdata_seqs, method='deeplift', aggregate_strand=True, batch_size=256)
    double_alt_imp_contrib = {k: v * dpdata_seqs for k, v in double_alt_imp_scores.items()}

    double_mut = NumpyDataset({t: {
        "pred": double_alt_preds[t],
        "imp": {
            "profile": double_alt_imp_contrib[f"{t}/weighted"],
            "count": double_alt_imp_contrib[f"{t}/count"],
        }} for t in tasks}, attrs={'index': 'motif_pair_idx'})

    # store all files to disk
    remove_exists(str(output_dir / 'ref.h5'), overwrite=True)
    ref.save(output_dir / 'ref.h5')

    remove_exists(str(output_dir / 'single_mut.h5'), overwrite=True)
    single_mut.save(output_dir / 'single_mut.h5')

    remove_exists(str(output_dir / 'double_mut.h5'), overwrite=True)
    double_mut.save(output_dir / 'double_mut.h5')

    dfi_subset.to_csv(output_dir / 'dfi_subset.csv.gz', compression='gzip')
    dfab.to_csv(output_dir / 'dfab.csv.gz', compression='gzip')
    return dfab, ref, single_mut, double_mut


def generate_motif_data(dfab, ref, single_mut, double_mut, pairs, output_dir, tasks,
                        profile_width=200, save=False,
                        pseudo_count_quantile=0.2,
                        profile_slice=slice(82, 119)):
    import gc
    from basepair.utils import write_pkl
    from basepair.exp.chipnexus.spacing import remove_edge_instances
    from basepair.exp.chipnexus.perturb.scores import (ism_compute_features_tidy,
                                                       compute_features_tidy,
                                                       SCORES,
                                                       max_profile_count)
    if save:
        c_output_dir = os.path.join(output_dir, 'motif_pair_lpdata')
        os.makedirs(c_output_dir, exist_ok=True)

    dfabf_ism_l = []
    dfabf_l = []
    for motif_pair in pairs:
        motif_pair_name = "<>".join(motif_pair)
        dfab_subset = remove_edge_instances(dfab[dfab.motif_pair == motif_pair_name], profile_width=profile_width)
        pdata = ParturbationDataset(dfab_subset, ref, single_mut, double_mut, profile_width=profile_width)
        output = pdata.load_all(num_workers=0)
        output['dfab'] = dfab_subset

        # Compute the directionality and epistasis scores
        # Epistasis:
        o = {motif_pair_name: output}
        dfabf_ism_l.append(ism_compute_features_tidy(o, tasks))
        # Directional:
        dfabf_l.append(compute_features_tidy(o, tasks, SCORES,
                                             pseudo_count_quantile=pseudo_count_quantile,
                                             profile_slice=profile_slice))

        # motif_pair_lpdata[motif_pair_name] = output
        # sort_idx = np.argsort(pdata.dfab.center_diff)
        if save:
            write_pkl(output, os.path.join(c_output_dir, motif_pair_name + '.pkl'))
        del o
        del output
        del pdata
        del dfab_subset
        print("Garbage collect")
        gc.collect()

    return pd.concat(dfabf_ism_l, axis=0), pd.concat(dfabf_l, axis=0)


# --------------------------------------------


def plot_scatter(x_ref, x_alt, y_ref, y_alt, ax, alpha=.2, s=1, label=None, xl=[0, 2], pval=True):
    ax.scatter(x_alt / x_ref,
               y_alt / y_ref, alpha=alpha, s=s, label=label)
    if pval:
        xpval = wilcoxon(x_ref, x_alt).pvalue
        ypval = wilcoxon(y_ref, y_alt).pvalue
        kwargs = dict(size="small", horizontalalignment='center')
        ax.text(1.8, 1, f"{xpval:.2g}", **kwargs)
        ax.text(1, 1.8, f"{ypval:.2g}", **kwargs)
    alpha = .5
    ax.plot(xl, xl, c='grey', alpha=alpha)
    ax.axvline(1, c='grey', alpha=alpha)
    ax.axhline(1, c='grey', alpha=alpha)
    ax.set_xlim(xl)
    ax.set_ylim(xl)


def plt_diag(xl, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(xl)
    ax.set_ylim(xl)
    ax.plot(xl, xl, c='grey', alpha=0.5)


def compute_features(dfab_sm):
    return {"Corrected total counts": dict(
        x_alt=(dfab_sm.xy_alt_pred_inside + dfab_sm.xy_alt_pred_outside) - dfab_sm.dxy_y_pred_total,  # total counts | dB - dxy_x_pred_total
        x_ref=(dfab_sm.xy_ref_pred_inside + dfab_sm.xy_ref_pred_outside) - (dfab_sm.dy_y_alt_pred_inside + dfab_sm.dy_y_alt_pred_outside),
        y_alt=(dfab_sm.yx_alt_pred_inside + dfab_sm.yx_alt_pred_outside) - dfab_sm.dxy_x_pred_total,
        y_ref=(dfab_sm.yx_ref_pred_inside + dfab_sm.yx_ref_pred_outside) - (dfab_sm.dx_x_alt_pred_inside + dfab_sm.dx_x_alt_pred_outside)
    ),
        "Corrected footprint counts": dict(
        # (A|dB - A|dA&dB)/(A - A|dA)
        x_alt=dfab_sm.xy_alt_pred_inside - dfab_sm.dxy_y_pred_inside,  # total counts | dB - dxy_x_pred_total
        x_ref=dfab_sm.xy_ref_pred_inside - dfab_sm.dy_y_alt_pred_inside,
        y_alt=dfab_sm.yx_alt_pred_inside - dfab_sm.dxy_x_pred_inside,
        y_ref=dfab_sm.yx_ref_pred_inside - dfab_sm.dx_x_alt_pred_inside
    ),
        "Total counts": dict(
        x_alt=(dfab_sm.xy_alt_pred_inside + dfab_sm.xy_alt_pred_outside),
        x_ref=(dfab_sm.xy_ref_pred_inside + dfab_sm.xy_ref_pred_outside),
        y_alt=(dfab_sm.yx_alt_pred_inside + dfab_sm.yx_alt_pred_outside),
        y_ref=(dfab_sm.yx_ref_pred_inside + dfab_sm.yx_ref_pred_outside)
    ),

        "Profile counts": dict(
        x_alt=dfab_sm.xy_alt_pred_inside,
        x_ref=dfab_sm.xy_ref_pred_inside,
        y_alt=dfab_sm.yx_alt_pred_inside,
        y_ref=dfab_sm.yx_ref_pred_inside
    ),
        "Rescaled profile importance": dict(
        x_alt=(dfab_sm.xy_alt_imp_inside * (dfab_sm.xy_alt_pred_inside + dfab_sm.xy_alt_pred_outside)),
        x_ref=(dfab_sm.xy_ref_imp_inside * (dfab_sm.xy_ref_pred_inside + dfab_sm.xy_ref_pred_outside)),
        y_alt=(dfab_sm.yx_alt_imp_inside * (dfab_sm.yx_alt_pred_inside + dfab_sm.yx_alt_pred_outside)),
        y_ref=(dfab_sm.yx_ref_imp_inside * (dfab_sm.yx_ref_pred_inside + dfab_sm.yx_ref_pred_outside))
    ),
        "Profile importance": dict(
        x_alt=dfab_sm.xy_alt_imp_inside,
        x_ref=dfab_sm.xy_ref_imp_inside,
        y_alt=dfab_sm.yx_alt_imp_inside,
        y_ref=dfab_sm.yx_ref_imp_inside
    ),
        "Count importance": dict(
        x_alt=dfab_sm.xy_alt_impcount_inside,
        x_ref=dfab_sm.xy_ref_impcount_inside,
        y_alt=dfab_sm.yx_alt_impcount_inside,
        y_ref=dfab_sm.yx_ref_impcount_inside
    ),
        "Profile match": dict(
        x_alt=dfab_sm.xy_alt_pred_match,
        x_ref=dfab_sm.xy_ref_pred_match,
        y_alt=dfab_sm.yx_alt_pred_match,
        y_ref=dfab_sm.yx_ref_pred_match
    )}


def plot_pairs(dfab_pairs, pairs, plot_features, pseudo_count_quantile=0, variable=None, pval=False):
    from basepair.plot.config import get_figsize
    nf = len(plot_features)
    fig, axes = plt.subplots(nrows=len(pairs), ncols=nf, figsize=get_figsize(1 / 3 * nf, len(pairs) / nf))
    for i, motif_pair in enumerate(pairs):
        k = "<>".join(motif_pair)
        dfab_sma = dfab_pairs[k]
        dfab_sma = dfab_sma[dfab_sma.center_diff < 150]

        cat_dist = pd.Categorical(pd.cut(dfab_sma.center_diff, [0, 35, 70, 150]))
        cat_strand = pd.Categorical(dfab_sma.strand_combination)

        match_threshold = .2
        cat_match = pd.Categorical(((dfab_sma.match_weighted_p_x > match_threshold).map({True: 'high', False: 'low'}) + "-" +
                                    (dfab_sma.match_weighted_p_y > .2).map({True: 'high', False: 'low'})))
        cat_imp = pd.Categorical(((dfab_sma.imp_weighted_p_x > match_threshold).map({True: 'high', False: 'low'}) + "-" +
                                  (dfab_sma.imp_weighted_p_y > .2).map({True: 'high', False: 'low'})))

        dfab_sm = dfab_sma
        for j, ax in enumerate(axes[i]):
            if i == 0:
                ax.set_title(plot_features[j])
            # Compute the pseudo-counts
            all_features = compute_features(dfab_sma)
            features = all_features[plot_features[j]]
            if pseudo_count_quantile == 0:
                pseudo_count_x = 0
                pseudo_count_y = 0
            else:
                pseudo_count_x = np.percentile(features['x_ref'], 100 * pseudo_count_quantile)
                pseudo_count_y = np.percentile(features['y_ref'], 100 * pseudo_count_quantile)

            if variable is not None:
                for k, cat in enumerate(eval(variable).categories):
                    dfab_sm = dfab_sma[eval(variable) == cat]
                    all_features = compute_features(dfab_sm)
                    features = all_features[plot_features[j]]
                    plot_scatter(features['x_ref'] + pseudo_count_x, features['x_alt'] + pseudo_count_x,
                                 features['y_ref'] + pseudo_count_y, features['y_alt'] + pseudo_count_y,
                                 ax, alpha=.2, s=1, label=cat, pval=False)
            else:
                plot_scatter(features['x_ref'] + pseudo_count_x, features['x_alt'] + pseudo_count_x,
                             features['y_ref'] + pseudo_count_y, features['y_alt'] + pseudo_count_y,
                             ax, alpha=.2, s=1, label=None, pval=pval)
            ax.set_xlabel(r"${}\;(\Delta {})$".format(motif_pair[1], motif_pair[0]))
            ax.set_ylabel(r"${}\;(\Delta {})$".format(motif_pair[0], motif_pair[1]))
            if j == nf - 1 and i == 0 and variable is not None:
                ax.legend(scatterpoints=1, ncol=2, markerscale=10, columnspacing=0, loc='upper right',
                          handletextpad=0, borderpad=0, frameon=False, title=variable)
    plt.tight_layout()


def plot_mutation_heatmap(dfab_pairs, pairs, motif_list, feature='Corrected footprint counts',
                          signif_threshold=1e-5, ax=None, max_frac=2):
    import seaborn as sns
    if ax is None:
        ax = plt.gca()

    motifs = motif_list
    motif_to_idx = {m: i for i, m in enumerate(motifs)}

    o = np.zeros((len(motifs), len(motifs)))
    op = np.zeros((len(motifs), len(motifs)))

    for motif_pair in pairs:
        i, j = motif_to_idx[motif_pair[0]], motif_to_idx[motif_pair[1]]
        dfab_sma = dfab_pairs["<>".join(motif_pair)]
        dfab_sma = dfab_sma[(dfab_sma.center_diff < 150)]
        features = compute_features(dfab_sma)[feature]

        o[i, j] = np.mean(features['y_alt'] / features['y_ref'])  # x|dy
        o[j, i] = np.mean(features['x_alt'] / features['x_ref'])  # y|dx
        op[i, j] = wilcoxon(features['y_ref'], features['y_alt']).pvalue
        op[i, j] = wilcoxon(features['x_ref'], features['x_alt']).pvalue

    signif = op < signif_threshold
    a = np.zeros_like(signif).astype(str)
    a[signif] = "*"
    a[~signif] = ""

    sns.heatmap(pd.DataFrame(o, columns=["d" + x for x in motifs], index=motifs),
                annot=a, fmt="", vmin=2 - max_frac, vmax=max_frac,
                cmap='RdBu_r', ax=ax)
    ax.set_title(f"{feature} (alt / ref) (*: p<{signif_threshold})")
