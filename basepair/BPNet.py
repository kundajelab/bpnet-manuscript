import os
from kipoi.data_utils import get_dataset_item, numpy_collate_concat
import keras.backend as K
import matplotlib.ticker as ticker
from basepair.functions import softmax
from genomelake.extractors import FastaExtractor
from keras.models import load_model
from collections import OrderedDict
from basepair.plot.tracks import plot_tracks, filter_tracks
from basepair.extractors import extract_seq
from basepair.data import numpy_minibatch, nested_numpy_minibatch
from basepair.losses import MultichannelMultinomialNLL, mc_multinomial_nll_2, mc_multinomial_nll_1, twochannel_multinomial_nll
from tqdm import tqdm
from basepair.utils import flatten_list
from kipoi.utils import unique_list
from concise.utils.plot import seqlogo
from basepair.functions import mean
from concise.preprocessing import encodeDNA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from genomelake.extractors import BigwigExtractor
import pyBigWig
from pysam import FastaFile
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def model2tasks(model):
    """Get tasks from the model
    """
    return unique_list([l.name.split("/")[1].split("_")[0] for l in model.outputs])


def model2seqlen(model):
    """Get the sequence length from the model
    """
    return model.layers[0].output.shape[1].value


class BPNet:

    def __init__(self, model, tasks=None, fasta_file=None, preproc=None, bias_model=None):
        """Main interface to BPNet

        Args:
          model: Keras model
          tasks: if None, it will be inferred from the model
          tasks: list of task names
          fasta_file: fasta_file path
        """
        self.model = model
        self.fasta_file = fasta_file
        if tasks is None:
            self.tasks = model2tasks(model)
        else:
            self.tasks = tasks
        self.preproc = preproc
        self.bias_model = bias_model

        self.grad_fns = {}

    @classmethod
    def train(cls,
              dataspec,
              output_dir=".",
              hparams=None,
              gpu=0,
              num_workers=10,
              touch_files=False,
              hp_train="",
              hp_model="",
              cometml_project="",
              render_report=False,
              report_template="",
              cache_data=False):
        """Create a new bpnet model by training it

        Args:
          data_spec: Dataset specification file
          output_dir: output directory
          hparams: (optional) File path to the hyper-parameters yaml file for model-training
          cache_dir: (optional) if set, the data function will get cached
          hp_train: override hypter-parameter settings for hp.train.kwargs.
            use 'k1=v1,k2=v2' synthax
          hp_models: override hypter-parameter settings for hp.model.kwargs
          cometml_project: comet_ml project name. Example: Avsecz/basepair.
            If not specified, cometml will not get used
          render_report (bool): if True, render the report by executing a ipynb
          report_template (str): template to use for the report path to the ipython notebook template
          gpu: (optional) which gpu to use

        dataspec.yaml
        ---------------
            task_specs:
              <task_id>:
                pos_counts: counts.pos.bw
                neg_counts: counts.neg.bw
                peaks: peaks.bed
                ignore_strand: False
            fasta_file: my_genome.fa
            peaks: peak.bed

        hparams.yaml
        ------------
            data:
              peak_width: 200
              shuffle: true
              valid_chr:
                - chr2
                - chr3
                - chr4
              test_chr:
                - chr1
                - chr8
                - chr9
            model:
              name: seq_multitask
              kwargs:
                filters: 21
                conv1_kernel_size: 21
                tconv_kernel_size: 25
                n_dil_layers: 6
                lr: 0.004
                c_task_weight: 100
            train:
              early_stop_patience: 5
              epochs: 200
              batch_size: 2
        """
        from basepair.cli.train import train
        train(dataspec=dataspec,
              output_dir=output_dir,
              hparams=hparams,
              gpu=gpu,
              num_workers=num_workers,
              touch_files=touch_files,
              hp_train=hp_train,
              hp_model=hp_model,
              cometml_project=cometml_project,
              render_report=render_report,
              report_template=report_template,
              cache_data=cache_data)
        return cls.from_mdir(output_dir)

    def input_seqlen(self):
        """Get the sequence length from the model
        """
        return self.model.layers[0].output.shape[1].value

    def _get_input_tensor(self):
        return self.model.inputs[0]  # always compute importance scores w.r.t. the sequence

    def _get_output_tensor(self, model_outputs, strand_id, task_id, pred_summary):
        if pred_summary == 'count':
            return model_outputs[len(self.tasks) + task_id][:, strand_id]
        elif pred_summary == 'weighted':
            raw_outputs = model_outputs[task_id][:, :, strand_id]
            return K.sum(K.stop_gradient(K.softmax(raw_outputs)) * raw_outputs, axis=-1)
        elif pred_summary == 'l2':
            raw_outputs = model_outputs[task_id][:, :, strand_id]
            return K.sum(raw_outputs[:, :, strand_id] * raw_outputs[:, :, strand_id], axis=-1)
        elif pred_summary == 'l2_pre-act':
            raw_outputs = model_outputs[task_id][:, :, strand_id]
            return K.sum(K.log(raw_outputs[:, :, strand_id]) *
                         K.log(raw_outputs[:, :, strand_id]), axis=-1)
        # elif pred_summary == 'max':
        # elif pred_summary == 'mean':
        else:
            ValueError(f"pred_summary={pred_summary} couldn't be interpreted")

    def _imp_grad_fn(self, strand, task_id, pred_summary):
        k = f"grad/{strand}/{task_id}/{pred_summary}"
        if k in self.grad_fns:
            return self.grad_fns[k]
        # Actually compute it
        strand_id = {"pos": 0, "neg": 1}[strand]
        inp = self._get_input_tensor()

        fn = K.function(self.model.inputs,
                        K.gradients(self._get_output_tensor(self.model.outputs, strand_id,
                                                            task_id, pred_summary), inp))
        self.grad_fns[k] = fn
        return fn

    def _imp_ism_fn(self, strand, task_id, pred_summary):
        k = f"ism/{strand}/{task_id}/{pred_summary}"
        if k in self.grad_fns:
            return self.grad_fns[k]
        # Actually compute it
        strand_id = {"pos": 0, "neg": 1}[strand]
        keras_model = self.model

        def get_ism_score(onehot_data):
            from collections import OrderedDict
            from basepair.functions import softmax

            # create mutations
            onehot_data = np.array(onehot_data)[0]
            # print("one hot data: "+str(np.array(onehot_data).shape))
            mutated_seqs = []
            for sample in onehot_data:
                # print("sample: "+str(np.array(sample).shape))
                for pos in range(len(sample)):
                    for base in range(4):
                        mutated = sample.copy()
                        mutated[pos] = np.zeros((1, 4))
                        mutated[pos][base] = 1
                        mutated_seqs.append(mutated)
            mutated_seqs = np.array(mutated_seqs)
            # print("mutated seqs: "+str(mutated_seqs.shape))

            # get predictions
            raw_predictions = keras_model.predict(mutated_seqs, batch_size=32)

            # get scores
            attribs = []
            for sample_idx, sample in enumerate(onehot_data):
                temp_attribs = []
                for pos in range(len(sample)):
                    temp_attribs.append([])
                    for base in range(4):
                        if pred_summary == 'count':
                            relevant_output = raw_predictions[len(self.tasks) + task_id][(sample_idx * len(sample) * 4) + (pos * 4) + base]
                            temp_attribs[pos].append(relevant_output[strand_id])
                        else:
                            relevant_output = raw_predictions[task_id][(sample_idx * len(sample) * 4) + (pos * 4) + base]
                            temp_attribs[pos].append(np.sum(softmax([relevant_output[:, strand_id]]) * [relevant_output[:, strand_id]]))
                temp_attribs = np.array(temp_attribs)
                avg_scores = np.mean(temp_attribs, axis=1, keepdims=True)  # this is ACGT axis
                temp_attribs -= avg_scores
                # print("sample attribs: "+str(temp_attribs.shape))
                attribs.append(temp_attribs)
            attribs = np.array([attribs])
            # print("all attribs: "+str(attribs.shape))
            # attribs = np.swapaxes(np.array(attribs), -1, -2)
            return attribs

        fn = get_ism_score
        self.grad_fns[k] = fn
        return fn

    def _imp_deeplift_fn(self, x, strand, task_id, pred_summary):
        k = f"deeplift/{strand}/{task_id}/{pred_summary}"
        if k in self.grad_fns:
            return self.grad_fns[k]

        import deepexplain
        from deepexplain.tensorflow.methods import DeepLIFTRescale
        from deepexplain.tensorflow import DeepExplain
        from deeplift.dinuc_shuffle import dinuc_shuffle
        from collections import OrderedDict
        from keras.models import load_model, Model
        import keras.backend as K
        import numpy as np
        import tempfile
        print("Loading model...")
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp:
            self.model.save(temp.name)
            K.clear_session()
            self.grad_fns = {}
            self.model = load_model(temp.name)

        with deepexplain.tensorflow.DeepExplain(session=K.get_session()) as de:
            input_tensor = self._get_input_tensor()
            fModel = Model(inputs=input_tensor, outputs=self.model.outputs)
            for strand_id, strand in enumerate(["pos", "neg"]):
                for task_id in range(len(self.tasks)):
                    for pred_summary in ['weighted', 'count']:
                        k_tmp = f"deeplift/{strand}/{task_id}/{pred_summary}"
                        target_tensor = self._get_output_tensor(fModel(input_tensor), strand_id, task_id, pred_summary)
                        self.grad_fns[k_tmp] = de.explain('deeplift', target_tensor,
                                                          input_tensor, x)  # [:1]

        return self.grad_fns[k]

    def imp_score(self, x, task, strand='both', method='grad', pred_summary='weighted', batch_size=512):
        """Compute the importance score

        Args:
          x: one-hot encoded DNA sequence
          method: which importance score to use. Available: grad, ism, deeplift
          strand: for which strand to run it ('pos', 'neg' or 'both'). If None, the average of both strands is returned
          task_id: id of the task as an int. See `self.tasks` for available tasks

        """
        assert task in self.tasks
        # figure out the task id
        task_id = [i for i, t in enumerate(self.tasks) if t == task][0]

        # task_id
        if strand == 'both':
            # average across strands
            return mean([self.imp_score(x, task,
                                        strand=strand,
                                        method=method,
                                        pred_summary=pred_summary,
                                        batch_size=batch_size)
                         for strand in ['pos', 'neg']])

        def input_to_list(input_names, x):
            if isinstance(x, list):
                return x
            elif isinstance(x, dict):
                return [x[k] for k in input_names]
            else:
                return [x]

        input_names = self.model.input_names
        assert input_names[0] == "seq"

        # get the importance scoring function
        # if method == "grad":
        #     fn = self._imp_grad_fn(strand, task_id, pred_summary) #returns fxn
        #     fn_applied = fn(input_to_list(input_names, x))[0]
        # elif method == "ism":
        #     fn = self._imp_ism_fn(strand, task_id, pred_summary) #returns fxn
        #     fn_applied = fn(input_to_list(input_names, x))[0]
        # elif method == "deeplift":
        #     fn = self._imp_deeplift_fn(x, strand, task_id, pred_summary) #returns numpy.ndarray
        #     fn_applied = fn
        # else:
        #     raise ValueError("Please provide a valid importance scoring method: grad, ism or deeplift")

        # if batch_size is None:
        #     return fn_applied
        # else:
        #     return numpy_collate_concat([fn_applied for batch in nested_numpy_minibatch(x, batch_size=batch_size)])

        if method == "grad":
            fn = self._imp_grad_fn(strand, task_id, pred_summary)
        elif method == "ism":
            fn = self._imp_ism_fn(strand, task_id, pred_summary)
        elif method == "deeplift":
            fn = self._imp_deeplift_fn(x, strand, task_id, pred_summary)
        else:
            raise ValueError("Please provide a valid importance scoring method: grad, ism or deeplift")

        if batch_size is None:
            return fn(input_to_list(input_names, x))[0]
        else:
            return numpy_collate_concat([fn(input_to_list(input_names, batch))[0]
                                         for batch in nested_numpy_minibatch(x, batch_size=batch_size)])

    def imp_score_all(self, seq, method='grad', aggregate_strand=False, batch_size=512,
                      pred_summaries=['weighted', 'count']):
        """Compute all importance scores

        Args:
          seq: one-hot encoded DNA sequences
          method: 'grad', 'deeplift' or 'ism'
          aggregate_strands: if True, the average importance scores across strands will be returned
          batch_size: batch size when computing the importance scores

        Returns:
          dictionary with keys: {task}/{pred_summary}/{strand_i} or {task}/{pred_summary}
          and values with the same shape as `seq` corresponding to importance scores
        """
        d_n_channels = {task: 2 for task_id, task in enumerate(self.tasks)}
        # TODO - update
        # preds_dict['counts'][task_id].shape[-1]

        # TODO - implement the ism version
        # if method == 'ism':
        #     return self.ism()

        out = {f"{task}/{pred_summary}/{strand_i}": self.imp_score(seq,
                                                                   task=task,
                                                                   strand=strand,
                                                                   method=method,
                                                                   pred_summary=pred_summary,
                                                                   batch_size=batch_size)
               for task in self.tasks
               for strand_i, strand in enumerate(['pos', 'neg'][:d_n_channels[task]])
               for pred_summary in pred_summaries}
        if aggregate_strand:
            return {f"{task}/{pred_summary}": mean([out[f"{task}/{pred_summary}/{strand_i}"]
                                                    for strand_i, strand in enumerate(['pos', 'neg'][:d_n_channels[task]])])
                    for pred_summary in ['weighted', 'count']
                    for task in self.tasks}
        else:
            return out

    def get_seq(self, intervals, variants=None, use_strand=False):
        """Get the one-hot-encoded sequence used to make model predictions and
        optionally augment it with the variants
        """
        if variants is not None:
            if use_strand:
                raise NotImplementedError("use_strand=True not implemented for variants")
            # Augment the intervals using a variant
            if not isinstance(variants, list):
                variants = [variants] * len(intervals)
            else:
                assert len(variants) == len(intervals)
            seq = np.stack([extract_seq(interval, variant, self.fasta_file, one_hot=True)
                            for variant, interval in zip(variants, intervals)])
        else:
            variants = [None] * len(intervals)
            seq = FastaExtractor(self.fasta_file, use_strand=use_strand)(intervals)
        return seq

    def sim_pred(self, central_motif, side_motif=None, side_distances=[], repeat=128, importance=[]):
        """
        Args:
          importance: list of importance scores
        """
        # TODO - update?
        from basepair.exp.chipnexus.simulate import generate_seq, postproc, average_profiles, flatten
        batch_size = repeat
        seqlen = self.input_seqlen()
        tasks = self.tasks

        # simulate sequence
        seqs = encodeDNA([generate_seq(central_motif, side_motif=side_motif,
                                       side_distances=side_distances, seqlen=seqlen)
                          for i in range(repeat)])

        # get predictions
        preds = self.model.predict(seqs, batch_size=batch_size)
        # TODO - remove this and use model.predict instead
        scaled_preds = postproc(preds, tasks)

        if importance:
            # get the importance scores
            imp_scores = self.seq_importance(seqs, importance)

            # merge and aggregate the profiles
            out = {"imp": imp_scores, "profile": scaled_preds}
        else:
            out = scaled_preds
        return average_profiles(flatten(out, "/"))

    def predict(self, seq, batch_size=512):
        """Make model prediction

        Args:
          seq: numpy array of one-hot-encoded array of sequences
          batch_size: batch size

        Returns:
          dictionary key=task and value=prediction for the task
        """
        if self.bias_model is not None:
            # TODO - what is this?
            seq, = self.bias_model.predict((seq, ), batch_size)

        preds = self.model.predict(seq, batch_size=batch_size)

        if len(self.model.output) == 2 * len(self.tasks):
            # extract the profile and count predictions
            profile_preds = {task: softmax(preds[task_i]) for task_i, task in enumerate(self.tasks)}
            count_preds = {task: preds[len(self.tasks) + task_i] for task_i, task in enumerate(self.tasks)}
            # compute the scaling factor
            if self.preproc is None:
                scales = {task: np.exp(count_preds[task]) - 1
                          for task in self.tasks}
            else:
                scales = {task: np.exp(self.preproc.objects[f'profile/{task}'].steps[1][1].inverse_transform(count_preds[task])) - 1
                          for task in self.tasks}

            # scaled profile (counts per base)
            return {task: profile_preds[task] * scales[task][:, np.newaxis] for task in self.tasks}
        else:
            return {task: preds[task_i] for task_i, task in enumerate(self.tasks)}

    def predict_all(self, seq, imp_method='grad', batch_size=512, pred_summaries=['weighted', 'count']):
        """Make model prediction based
        """
        if self.bias_model is not None:
            seq, = self.bias_model.predict((seq, ), batch_size)

        preds = self.predict(seq, batch_size=batch_size)

        if imp_method is not None:
            imp_scores = self.imp_score_all(seq, method=imp_method, aggregate_strand=True,
                                            batch_size=batch_size, pred_summaries=pred_summaries)
        else:
            imp_scores = dict()

        out = [dict(
            seq=get_dataset_item(seq, i),
            # interval=intervals[i],
            pred=get_dataset_item(preds, i),
            # TODO - shall we call it hyp_imp score or imp_score?
            imp_score=get_dataset_item(imp_scores, i),
        ) for i in range(len(seq))]
        return out

    def predict_intervals(self, intervals,
                          variants=None,
                          imp_method='grad',
                          use_strand=False,
                          batch_size=512):
        """
        Args:
          intervals: list of pybedtools.Interval
          variant: a single instance or a list bpnet.extractors.Variant
          pred_summary: 'mean' or 'max', summary function name for the profile gradients
          compute_grads: if False, skip computing gradients
        """
        # TODO - support also other importance scores
        seq = self.get_seq(intervals, variants, use_strand=use_strand)

        preds = self.predict_all(seq, imp_method, batch_size)

        # append intervals
        for i in range(len(seq)):
            preds[i]['interval'] = intervals[i]
            if variants is not None:
                preds[i]['variant'] = variants[i]
        return preds

    def plot_intervals(self, intervals, ds=None, variants=None,
                       seqlets=[],
                       pred_summary='weighted',
                       imp_method='grad',
                       batch_size=128,
                       # ylim=None,
                       xlim=None,
                       # seq_height=1,
                       rotate_y=0,
                       add_title=True,
                       fig_height_per_track=2,
                       same_ylim=False,
                       fig_width=20):
        """Plot predictions

        Args:
          intervals: list of pybedtools.Interval
          variant: a single instance or a list of bpnet.extractors.Variant
          ds: DataSpec. If provided, the ground truth will be added to the plot
          pred_summary: 'mean' or 'max', summary function name for the profile gradients
        """
        out = self.predict_intervals(intervals,
                                     variants=variants,
                                     imp_method=imp_method,
                                     # pred_summary=pred_summary,
                                     batch_size=batch_size)
        figs = []
        if xlim is None:
            xmin = 0
        else:
            xmin = xlim[0]
        shifted_seqlets = [s.shift(-xmin) for s in seqlets]

        for i in range(len(out)):
            pred = out[i]
            interval = out[i]['interval']

            if ds is not None:
                obs = {task: ds.task_specs[task].load_counts([interval])[0] for task in self.tasks}
            else:
                obs = None

            title = "{i.chrom}:{i.start}-{i.end}, {i.name} {v}".format(i=interval, v=pred.get('variant', ''))

            # handle the DNase case
            if isinstance(pred['seq'], dict):
                seq = pred['seq']['seq']
            else:
                seq = pred['seq']

            if obs is None:
                # TODO - simplify?
                viz_dict = OrderedDict(flatten_list([[
                    (f"{task} Pred", pred['pred'][task]),
                    (f"{task} Imp profile", pred['imp_score'][f"{task}/{pred_summary}"] * seq),
                    # (f"{task} Imp counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
                ] for task_idx, task in enumerate(self.tasks)]))
            else:
                viz_dict = OrderedDict(flatten_list([[
                    (f"{task} Pred", pred['pred'][task]),
                    (f"{task} Obs", obs[task]),
                    (f"{task} Imp profile", pred['imp_score'][f"{task}/{pred_summary}"] * seq),
                    # (f"{task} Imp counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
                ] for task_idx, task in enumerate(self.tasks)]))

            if add_title:
                title = "{i.chrom}:{i.start}-{i.end}, {i.name} {v}".format(i=interval, v=pred.get('variant', '')),
            else:
                title = None

            if same_ylim:
                fmax = {feature: max([np.abs(viz_dict[f"{task} {feature}"]).max() for task in self.tasks])
                        for feature in ['Pred', 'Imp profile', 'Obs']}

                ylim = []
                for k in viz_dict:
                    f = k.split(" ", 1)[1]
                    if "Imp" in f:
                        ylim.append((-fmax[f], fmax[f]))
                    else:
                        ylim.append((0, fmax[f]))
            else:
                ylim = None
            fig = plot_tracks(filter_tracks(viz_dict, xlim),
                              seqlets=shifted_seqlets,
                              title=title,
                              fig_height_per_track=fig_height_per_track,
                              rotate_y=rotate_y,
                              fig_width=fig_width,
                              ylim=ylim,
                              legend=True)
            figs.append(fig)
        return figs

    # TODO also allow imp_scores
    def export_bw(self,
                  intervals,
                  output_dir,
                  # pred_summary='weighted',
                  imp_method='grad',
                  batch_size=512,
                  scale_importance=False,
                  chromosomes=None):
        """Export predictions and model importances to big-wig files

        Args:
          intervals: list of genomic intervals
          output_dir: output directory

          batch_size:
          scale_importance: if True, multiple the importance scores by the predicted count value
          chromosomes: a list of chromosome names consisting a genome
        """
        #          pred_summary: which operation to use for the profile gradients
        logger.info("Get model predictions and importance scores")
        out = self.predict_intervals(intervals,
                                     imp_method=imp_method,
                                     batch_size=batch_size)

        logger.info("Setup bigWigs for writing")
        # Get the genome lengths
        fa = FastaFile(self.fasta_file)
        if chromosomes is None:
            genome = OrderedDict([(c, l) for c, l in zip(fa.references, fa.lengths)])
        else:
            genome = OrderedDict([(c, l) for c, l in zip(fa.references, fa.lengths) if c in chromosomes])
        fa.close()

        output_feats = ['preds.pos', 'preds.neg', 'importance.profile', 'importance.counts']

        # make sure the intervals are in the right order
        first_chr = list(np.unique(np.array([interval.chrom for interval in intervals])))
        last_chr = [c for c, l in genome.items() if c not in first_chr]
        genome = [(c, genome[c]) for c in first_chr + last_chr]

        # open bigWigs for writing
        bws = {}
        for task in self.tasks:
            bws[task] = {}
            for feat in output_feats:
                bw_preds_pos = pyBigWig.open(f"{output_dir}/{task}.{feat}.bw", "w")
                bw_preds_pos.addHeader(genome)
                bws[task][feat] = bw_preds_pos

        def add_entry(bw, arr, interval, start_idx=0):
            """Macro for adding an entry to the bigwig file

            Args:
              bw: pyBigWig file handle
              arr: 1-dimensional numpy array
              interval: genomic interval pybedtools.Interval
              start_idx: how many starting values in the array to skip
            """
            assert arr.ndim == 1
            assert start_idx < len(arr)

            if interval.stop - interval.start != len(arr):
                logger.error(f"interval.stop - interval.start ({interval.stop - interval.start})!= len(arr) ({len(arr)})")
                logger.error(f"Skipping the entry: {interval}")
                return
            bw.addEntries(interval.chrom, interval.start + start_idx,
                          values=arr[start_idx:],
                          span=1, step=1)

        # interval logic to handle overlapping intervals
        #   assumption: all intervals are sorted w.r.t the start coordinate
        #   strategy: don't write values at the same position twice (skip those)
        #
        # graphical representation:
        # ...     ]    - prev_stop
        #      [     ]   - new interval 1
        #         [  ]   - added chunk from interval 1
        #   [  ]         - new interval 2 - skip
        #          [   ] - new interval 3, fully add

        logger.info("Writing to bigWigs")
        prev_stop = None   # Keep track of what the previous interval already covered
        prev_chrom = None
        for i in tqdm(range(len(out))):
            interval = out[i]['interval']

            if prev_chrom != interval.chrom:
                # Encountered a new chromosome
                prev_stop = 0  # Restart the end-counter
                prev_chrom = interval.chrom

            if prev_stop >= interval.stop:
                # Nothing new to add to that range
                continue
            start_idx = max(prev_stop - interval.start, 0)

            for tid, task in enumerate(self.tasks):
                # Write predictions
                preds = out[i]['pred'][task]
                add_entry(bws[task]['preds.pos'], preds[:, 0],
                          interval, start_idx)
                add_entry(bws[task]['preds.neg'], preds[:, 1],
                          interval, start_idx)

                # Get the importance scores
                seq = out[i]['seq']
                hyp_imp = out[i]['imp_score']

                if scale_importance:
                    si_profile = preds.sum()  # Total number of counts in the region
                    si_counts = preds.sum()
                else:
                    si_profile = 1
                    si_counts = 1

                # profile - multipl
                add_entry(bws[task]['importance.profile'],
                          hyp_imp[f'{task}/weighted'][seq.astype(bool)] * si_profile,
                          interval, start_idx)
                add_entry(bws[task]['importance.counts'],
                          hyp_imp[f'{task}/count'][seq.astype(bool)] * si_counts,
                          interval, start_idx)

            prev_stop = max(interval.stop, prev_stop)

        logger.info("Done writing. Closing bigWigs")
        # Close all the big-wig files
        for task in self.tasks:
            for feat in output_feats:
                bws[task][feat].close()
        logger.info(f"Done! Files located at: {output_dir}")

    @classmethod
    def from_mdir(cls, model_dir):
        """
        Args:
          model_dir (str): Path to the model directory
        """
        import os
        from basepair.cli.schemas import DataSpec
        from keras.models import load_model
        from basepair.utils import read_pkl
        ds = DataSpec.load(os.path.join(model_dir, "dataspec.yaml"))
        model = load_model(os.path.join(model_dir, "model.h5"))
        preproc_file = os.path.join(model_dir, "preprocessor.pkl")
        if os.path.exists(preproc_file):
            preproc = read_pkl(preproc_file)
        else:
            preproc = None
        return cls(model=model, fasta_file=ds.fasta_file, tasks=list(ds.task_specs), preproc=preproc)

    # TODO - add `evaluate(intervals, bws)


class BPNetSeqModel(BPNet):
    """BPNet based on SeqModel
    """

    def __init__(self, seqmodel, fasta_file=None):
        self.seqmodel = seqmodel
        self.tasks = self.seqmodel.tasks
        self.fasta_file = fasta_file
        self.bias_model = None

    @classmethod
    def from_mdir(cls, model_dir):
        from basepair.seqmodel import SeqModel
        # TODO - figure out also the fasta_file if present (from dataspec)
        from basepair.cli.schemas import DataSpec
        ds_path = os.path.join(model_dir, "dataspec.yaml")
        if os.path.exists(ds_path):
            ds = DataSpec.load(ds_path)
            fasta_file = ds.fasta_file
        else:
            fasta_file = None
        return cls(SeqModel.from_mdir(model_dir), fasta_file=fasta_file)

    def input_seqlen(self):
        return self.seqmodel.seqlen

    def predict(self, seq, batch_size=512):
        """Make model prediction

        Args:
          seq: numpy array of one-hot-encoded array of sequences
          batch_size: batch size

        Returns:
          dictionary key=task and value=prediction for the task
        """

        preds = self.seqmodel.predict(seq, batch_size=batch_size)
        return {task: preds[f'{task}/profile'] * np.exp(preds[f'{task}/counts'][:, np.newaxis])
                for task in self.seqmodel.tasks}

    def imp_score_all(self, seq, method='deeplift', aggregate_strand=True, batch_size=512,
                      pred_summaries=['weighted', 'count']):
        """Compute all importance scores

        Args:
          seq: one-hot encoded DNA sequences
          method: 'grad', 'deeplift' or 'ism'
          aggregate_strands: if True, the average importance scores across strands will be returned
          batch_size: batch size when computing the importance scores

        Returns:
          dictionary with keys: {task}/{pred_summary}/{strand_i} or {task}/{pred_summary}
          and values with the same shape as `seq` corresponding to importance scores
        """
        assert aggregate_strand

        imp_scores = self.seqmodel.imp_score_all(seq, method=method)

        return {f"{task}/" + self._get_old_imp_score_name(pred_summary): imp_scores[f"{task}/{pred_summary}"]
                for task in self.seqmodel.tasks
                for pred_summary in ['profile/wn', 'counts/pre-act']}

    def _get_old_imp_score_name(self, s):
        s2s = {"profile/wn": 'weighted', 'counts/pre-act': 'count'}
        return s2s[s]

    def sim_pred(self, central_motif, side_motif=None, side_distances=[], repeat=128, importance=[]):
        """
        Args:
          importance: list of importance scores
        """
        from basepair.exp.chipnexus.simulate import generate_seq, average_profiles, flatten
        batch_size = repeat
        seqlen = self.seqmodel.seqlen
        tasks = self.seqmodel.tasks

        # simulate sequence
        seqs = encodeDNA([generate_seq(central_motif, side_motif=side_motif,
                                       side_distances=side_distances, seqlen=seqlen)
                          for i in range(repeat)])

        # get predictions
        scaled_preds = self.predict(seqs, batch_size=batch_size)

        if importance:
            # get the importance scores (compute only the profile and counts importance)
            imp_scores_all = self.seqmodel.imp_score_all(seqs, intp_pattern=['*/profile/wn', '*/counts/pre-act'])
            imp_scores = {t: {self._get_old_imp_score_name(imp_score_name): seqs * imp_scores_all[f'{t}/{imp_score_name}']
                              for imp_score_name in importance}
                          for t in tasks}

            # merge and aggregate the profiles
            out = {"imp": imp_scores, "profile": scaled_preds}
        else:
            out = {"profile": scaled_preds}
        return average_profiles(flatten(out, "/"))


# Backward compatibility
BPNetPredictor = BPNet
