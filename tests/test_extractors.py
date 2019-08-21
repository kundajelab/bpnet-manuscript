from basepair.extractors import extract_seq, Variant
from pybedtools import Interval
from pytest import fixture

FASTA_FILE = "tests/data/example.fa"


@fixture
def interval():
    return Interval("chr1", 1, 3)


@fixture
def variant():
    return Variant("chr1", 2, "C", "G")


def test_snp(interval, variant):
    assert extract_seq(Interval("chr1", 1, 3),
                       Variant("chr1", 2, "C", "G"),
                       FASTA_FILE) == "GG"


def test_del(interval, variant):
    assert extract_seq(Interval("chr1", 1, 4),
                       Variant("chr1", 2, "CG", "C"),
                       FASTA_FILE) == "CTC"


def test_ins(interval, variant):
    assert extract_seq(Interval("chr1", 1, 4),
                       Variant("chr1", 2, "C", "CT"),
                       FASTA_FILE) == "CTG"
