import pandas as pd
import numpy as np
from basepair.preproc import balance_class_weight


def test_balance():
    labels = pd.Series(["a", "a", "a", "b", "b"])
    out = balance_class_weight(labels)
    assert np.all(out == [2 / 3,
                          2 / 3,
                          2 / 3,
                          1,
                          1])
    df = pd.DataFrame({"label": labels, "w": out})
    assert np.all(df.groupby("label").sum() == labels.value_counts().min())
