import numpy as np

from basepair.modisco.core import Pattern
from pytest import raises


def test_pattern():
    p = Pattern("test",
                np.random.randn(10, 4),
                dict(t=np.random.randn(10, 4)),
                dict(t=np.random.randn(10, 4)))
    assert p.tasks() == ['t']
    print(p)
    p.get_consensus()
    p.seq = np.abs(p.seq)
    p.get_seq_ic()
    p.copy()
    p.copy()

    p.rc()
    len(p)

    with raises(Exception):
        p = Pattern("test",
                    np.random.randn(10, 4),
                    dict(t=np.random.randn(10, 4)),
                    dict(t=np.random.randn(11, 4)))

    with raises(Exception):
        p = Pattern("test",
                    np.random.randn(10, 4),
                    [5] * 10,
                    np.random.randn(10, 4))


def test_pad():
    a = np.zeros((20, 4))
    a[12] = 1
    a_short = a[5:15]
    assert a_short[7][0] == 1
    assert a[12][0] == 1
    assert a[13][0] == 0

    p = Pattern("test",
                a_short,
                dict(t=a_short),
                dict(t=a_short),
                dict(t=a))

    assert np.all(p.profile['t'] == a)
    assert p.len_profile() == 20

    assert p.pad(10).len_profile() == 20
    assert len(p.pad(10)) == 10

    assert len(p.pad(10)) == 10
    assert p.shift(-2).profile['t'][10][0] == 1
    assert p.shift(-2).contrib['t'][5][0] == 1
    pnew = p.shift(1).rc().shift(1).rc()
    assert np.allclose(p.seq, pnew.seq)
    assert np.allclose(p.contrib['t'], pnew.contrib['t'])
    assert np.allclose(p.profile['t'], pnew.profile['t'])


def dont_test_parse_hdf4():
    from basepair.modisco.results import ModiscoResult
    mr = ModiscoResult("/s/project/avsec/basepair/modisco/modisco.h5")
    mr.open()

    mr.f.ls()
    metacluster = "metacluster_0"
    pattern = "pattern_0"
    pattern_grp = mr.get_pattern_grp(metacluster, pattern)
    p = Pattern.from_hdf5_grp(pattern_grp, "m0_p0")
    pt = p.trim_seq_ic(0.08)
    assert len(pt) == len(pt.contrib['Klf4'])
    p = mr.get_pattern("metacluster_0/pattern_0")
    import matplotlib.pyplot as plt
    p.plot(kind='all')
    p.plot(kind=['seq', 'contrib/Klf4'])
    p.plot(kind='seq')

    plt.show()
    mr.plot_pattern("metacluster_0/pattern_0", kind=['seq', 'contrib/Klf4'])
