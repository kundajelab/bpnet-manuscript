"""
Global variables
"""
from pathlib import Path
valid_chr = ['chr2', 'chr3', 'chr4']
test_chr = ['chr1', 'chr8', 'chr9']

# All considered chromosomes
all_chr = ['chr1',
           'chr2',
           'chr3',
           'chr4',
           'chr5',
           'chr6',
           'chr7',
           'chr8',
           'chr9',
           'chr10',
           'chr11',
           'chr12',
           'chr13',
           'chr14',
           'chr15',
           'chr16',
           'chr17',
           'chr18',
           'chr19',
           'chr20',
           'chr21',
           'chr22',
           'chrX',
           'chrY']

# google drive directory path to figures
GDRIVE_CHIPNEXUS_RAW_ID = "1oJDXr2vbP89WS4Dp3mYli-aJi7Q1oZbR"


def get_repo_root():
    """Returns the data directory
    """
    import inspect
    import os
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_path = os.path.dirname(os.path.abspath(filename))
    return Path(os.path.abspath(os.path.join(this_path, "..")))


def get_data_dir():
    """Returns the data directory
    """
    import inspect
    import os
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_path = os.path.dirname(os.path.abspath(filename))
    DATA = os.path.join(this_path, "../data")
    if not os.path.exists(DATA):
        raise ValueError(DATA + " folder doesn't exist")
    return os.path.abspath(DATA)


def create_tf_session(visiblegpus, per_process_gpu_memory_fraction=0.45):
    import os
    import tensorflow as tf
    import keras.backend as K
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    # session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    session = tf.Session(config=session_config)
    K.set_session(session)
    return session


def get_exp_dir(exp):
    """Get the experiment directory
    """
    ddir = get_data_dir()
    url = "http://mitra.stanford.edu/kundaje/avsec/chipnexus/"
    www_path = f"exp/chipnexus/{exp}"
    return f"{ddir}/processed/chipnexus/exp/{exp}", \
        f"{ddir}/www/{www_path}", \
        f"{url}/{www_path}"
