# 모듈 import
from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import re

import numpy as np
import pandas as pd

from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

import collections
from tqdm.notebook import tqdm
from collections import Counter

import scipy.sparse as spr
import pickle
import distutils.dir_util
import io
import sentencepiece as spm
from ast import literal_eval
from itertools import compress
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors


# 데이터 처리 함수들
def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]


def before_updt_date(cand_song_idx, updt_date, song_meta):
    updt_date = int(re.sub('-', '', updt_date)[:8])
    return_idx = []

    for i in cand_song_idx:
        if int(song_meta.loc[i, 'issue_date']) > updt_date:
            continue
        else:
            return_idx.append(i)

    return return_idx

def clean(x):
    return literal_eval(x)

# main

# 데이터 읽기
song_meta = pd.read_json('song_meta_fill_gnr.json', typ = 'frame', encoding="utf-8")
train = pd.read_json('train.json',typ='frame', encoding="utf8")
test = pd.read_json('test.json',typ='frame', encoding="utf8")
genre_gn_all = pd.read_json('genre_gn_all.json',typ='series',encoding='utf-8')

from matrix_factorization import MakeBaselineResults
FILE_PATH = ''
base = MakeBaselineResults(FILE_PATH)
results = base.run()

from cosine_filtering import PlaylistEmbedding
FILE_PATH = 'D:\mydev\insight\insight_project\kakao_arena\melon\\'
U_space = PlaylistEmbedding(FILE_PATH)
U_space.run()