"""
Export prediction to a bigWig file
"""

from pybedtools import BedTool
import pandas as pd
import os
from basepair.BPNet import BPNet, BPNetSeqModel
from basepair.utils import read_pkl
from basepair.config import create_tf_session
from basepair.cli.schemas import DataSpec
from basepair.cli.evaluate import load_data
from basepair.preproc import resize_interval
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def export_bw_workflow(model_dir,
                       bed_file,
                       output_dir,
                       imp_method='grad',
                       # pred_summary='weighted',
                       batch_size=512,
                       seqmodel=False,
                       scale_importance=False,
                       gpu=0):
    """
    Export model predictions to big-wig files

    Args:
      model_dir: model directory path
      output_dir: output directory path
      bed_file: file path to a bed-file containing
        the intervals of interest

    """
    # pred_summary: 'mean' or 'max', summary function name for the profile gradients
    os.makedirs(output_dir, exist_ok=True)
    if gpu is not None:
        create_tf_session(gpu)

    logger.info("Load model, preprocessor and data specs")
    if seqmodel:
        bp = BPNetSeqModel.from_mdir(model_dir)
    else:
        bp = BPNet.from_mdir(model_dir)

    seqlen = bp.input_seqlen()
    logger.info(f"Resizing intervals (fix=center) to model's input width of: {seqlen}")
    intervals = list(BedTool(bed_file))
    intervals = [resize_interval(interval, seqlen) for interval in intervals]
    logger.info("Sort the bed file")
    intervals = list(BedTool(intervals).sort())

    bp.export_bw(intervals=intervals,
                 output_dir=output_dir,
                 # pred_summary=pred_summary,
                 imp_method=imp_method,
                 batch_size=batch_size,
                 scale_importance=scale_importance,
                 chromosomes=None)  # infer chromosomes from the fasta file
