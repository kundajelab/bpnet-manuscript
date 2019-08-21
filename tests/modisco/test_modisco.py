"""Test modisco
"""

import pandas as pd
import h5py
import pytest
# TODO - load data


@pytest.fixture
def scores():
    """modisco result
    """
    "tests/data/modisco"
    with h5py.File("tests/data/modisco/scores.h5", "r") as f:
        list(f.keys())
        tasks = list(f['contrib_scores'].keys())

        contrib_scores = {task: f['contrib_scores'][task][:] for task in tasks}
        hyp_contrib_scores = {task: f['hyp_contrib_scores'][task][:] for task in tasks}
        one_hot = f['one_hot'][:]
    return tasks, one_hot, contrib_scores, hyp_contrib_scores


@pytest.fixture
def mr():
    from modisco.tfmodisco_workflow import workflow

    import modisco

    tasks, one_hot, contrib_scores, hyp_contrib_scores = scores()  # TODO - use as fixture

    track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
        task_names=tasks,
        contrib_scores=contrib_scores,
        hypothetical_contribs=hyp_contrib_scores,
        one_hot=one_hot)

    with h5py.File("tests/data/modisco/results.hdf5", "r") as grp:
        mr = workflow.TfModiscoResults.from_hdf5(grp, track_set=track_set)
    return mr


# mr = mr()
# scores = scores()


def test_find_instances(mr, scores):

    from basepair.modisco import find_instances, labelled_seqlets2df

    tasks, one_hot, contrib_scores, hyp_contrib_scores = scores
    seqlets = find_instances(mr, tasks, contrib_scores, hyp_contrib_scores, one_hot)
    assert len(seqlets) == 1760

    seqlet = seqlets[0]

    assert seqlet.pattern == 0
    assert seqlet.metacluster == 0
    assert seqlet.score_result.revcomp == 0.0

    # convert the seqlets to dataframe
    df = labelled_seqlets2df(seqlets)

    assert len(df) == len(seqlets)
    assert isinstance(df, pd.DataFrame)

    assert "example_idx" in df
    assert "metacluster" in df
    assert "pattern" in df
