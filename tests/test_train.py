"""
Test model training
"""
from basepair.cli.schemas import (TaskSpec, DataSpec, HParams,
                                  TrainHParams, DataHParams, ModelHParams)
from basepair.cli.evaluate import evaluate
from kipoi.utils import cd as cdir
from pytest import fixture

ddir = "tests/data"


@fixture
def dataspec():
    return DataSpec(
        bigwigs={"task1": TaskSpec(
            task='task1',
            pos_counts=f"{ddir}/pos.bw",
            neg_counts=f"{ddir}/neg.bw",
        ), "task2": TaskSpec(
            task='task2',
            pos_counts=f"{ddir}/pos.bw",
            neg_counts=f"{ddir}/neg.bw",
        )},
        fasta_file=f"{ddir}/ref.fa",
        peaks=f"{ddir}/peaks.bed"
    )


@fixture
def hparams():
    return HParams(
        train=TrainHParams(epochs=2,
                           batch_size=2),
        data=DataHParams(
            valid_chr=['chr1'],
            test_chr=[],
            peak_width=10))


# hparams = hparams()
# dataspec = dataspec()


def test_minimum(dataspec, hparams, tmpdir):
    tmpdir = str(tmpdir.mkdir("from_config"))
    train(dataspec, output_dir=tmpdir,
          hparams=hparams, gpu=None)


def test_file(tmpdir):
    with cdir("tests/data"):
        tmpdir = str(tmpdir.mkdir("from_file"))
        train("dataspec.yaml", output_dir=tmpdir,
              hparams="hparams.yaml", gpu=None)
    evaluate(tmpdir, splits=['train', 'valid'])
