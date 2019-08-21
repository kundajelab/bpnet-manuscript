import pandas as pd
import numpy as np
import json
from basepair.modisco.utils import shorten_pattern


def parse_json(data):
    try:
        return json.loads(data)
    except Exception:
        return {'klf4': None, 'nanog': None, 'oct4': None, 'sox2': None}


def read_annotated_csv(path, suffix='/l'):
    df_periodicity = pd.read_csv(path)

    dfl = df_periodicity.join(df_periodicity['Label'].apply(parse_json).apply(pd.Series))[['External ID', 'klf4', 'nanog', 'oct4', 'sox2']]
    dfl.columns = ['pattern_long', 'Klf4' + suffix, 'Nanog' + suffix, 'Oct4' + suffix, 'Sox2' + suffix]
    dfl['pattern'] = dfl.pattern_long.map(shorten_pattern)
    del dfl['pattern_long']
    return dfl
