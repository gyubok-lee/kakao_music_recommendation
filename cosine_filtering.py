"""
word2vec으로 구현한 코사인유사도 협업필터링
res는 행렬분해 모듈로 구현한 베이스라인
"""

# 모듈 import
import os
import json
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models import KeyedVectors
import distutils.dir_util
import io
from ast import literal_eval
import numpy as np
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

# word2vec
class PlaylistEmbedding:
    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH
        self.min_count = 3
        self.size = 100
        self.window = 210
        self.sg = 5
        self.p2v_model = WordEmbeddingsKeyedVectors(self.size)  # KeyedVectors.load("p2v_model2.model")
        self.w2v_model = Word2Vec.load("word2vecf.model")
        with open(os.path.join(FILE_PATH, 'train.json'), encoding="utf-8") as f:
            self.train = json.load(f)
        with open(os.path.join(FILE_PATH, 'test.json'), encoding="utf-8") as f:
            self.test = json.load(f)
        with open(os.path.join(FILE_PATH, 'val.json'), encoding="utf-8") as f:
            self.val = json.load(f)
        with open(os.path.join(FILE_PATH, 'res.json'), encoding="utf-8") as f:
            self.most_results = json.load(f)

    # 전체 데이터에서 곡과 태그를 사전형식으로 저장
    def get_dic(self, train, val):
        song_dic = {}
        tag_dic = {}
        train = train + val
        data = train + self.test
        for q in tqdm(data):
            song_dic[str(q['id'])] = q['songs']
            tag_dic[str(q['id'])] = q['tags']
        self.song_dic = song_dic
        self.tag_dic = tag_dic

        # total = [['songs1'],['songs2'],['songs3'], ...['tags1'],['tags2'],[tags3]...]
        total = list(map(lambda x: list(map(str, x['songs'])) + list(x['tags']), data))
        total = [x for x in total if len(x) > 1]
        self.total = total

    def get_w2v(self, total, min_count, size, window, sg):
        w2v_model = Word2Vec(total, min_count=min_count, size=size, window=window, sg=sg, workers=4)
        self.w2v_model = w2v_model
        w2v_model.save("word2vecf.model")

    # 플레이리스트의 벡터값을 산출
    def update_p2v(self, train, val, w2v_model):
        ID = []
        vec = []
        for q in tqdm(train + val):
            tmp_vec = 0
            if len(q['songs']) >= 1:
                for song in q['songs'] + q['tags']:
                    try:
                        tmp_vec += w2v_model.wv.get_vector(str(song))
                    except KeyError:
                        pass
            if type(tmp_vec) != int:
                ID.append(str(q['id']))
                vec.append(tmp_vec)
        self.p2v_model.add(ID, vec)
        self.p2v_model.save('p2v_modelf.model')

    # 가장 비슷한 플레이리스트의 노래와 태그를 추천
    def get_result(self, p2v_model, song_dic, tag_dic, most_results, val):
        answers = []
        for n, q in tqdm(enumerate(val), total=len(val)):
            try:
                most_id = [x[0] for x in p2v_model.most_similar(str(q['id']), topn=200)]
                get_song = []
                get_tag = []
                for ID in most_id:
                    get_song += song_dic[ID]
                    get_tag += tag_dic[ID]
                cand_song += list(pd.value_counts(get_song)[:1000].index)

                get_song = before_updt_date(cand_song, q['updt_date'], self.song_meta)  # before_updt_date 모듈 추가!
                get_tag = list(pd.value_counts(get_tag)[:20].index)
                answers.append({
                    "id": q["id"],
                    "songs": remove_seen(q["songs"], get_song)[:100],
                    "tags": remove_seen(q["tags"], get_tag)[:10],
                })
            except:
                answers.append({
                    "id": most_results[n]["id"],
                    "songs": most_results[n]['songs'],
                    "tags": most_results[n]["tags"],
                })
                # check and update answer
        for n, q in enumerate(answers):
            if len(q['songs']) != 100:
                answers[n]['songs'] += remove_seen(q['songs'], self.most_results[n]['songs'])[:100 - len(q['songs'])]
            if len(q['tags']) != 10:
                answers[n]['tags'] += remove_seen(q['tags'], self.most_results[n]['tags'])[:10 - len(q['tags'])]
        self.answers = answers

    def run(self):
        self.get_dic(self.train, self.val)
        # self.get_w2v(self.total, self.min_count, self.size, self.window, self.sg)
        self.update_p2v(self.train, self.test, self.w2v_model)
        self.get_result(self.p2v_model, self.song_dic, self.tag_dic, self.most_results, self.test)

        write_json(self.answers, 'results.json')

# 실행
FILE_PATH = 'D:\mydev\insight\insight_project\kakao_arena\melon\\'
U_space = PlaylistEmbedding(FILE_PATH)
U_space.run()