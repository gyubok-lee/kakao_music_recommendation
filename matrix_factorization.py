class MakeBaselineResults:

    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH

        with open(os.path.join(FILE_PATH, 'train_addtags.json'), encoding="utf-8") as f:
            self.train = pd.DataFrame(json.load(f))
            train.head()

        with open(os.path.join(FILE_PATH, 'val_addtags.json'), encoding="utf-8") as f:
            self.val = pd.DataFrame(json.load(f))

        self.train = pd.concat([self.train, self.val], ignore_index=True)  # val 데이터도 합쳐서 train

        with open(os.path.join(FILE_PATH, 'test_addtags.json'), encoding="utf-8") as f:
            self.test = pd.DataFrame(json.load(f))

        with open(os.path.join(FILE_PATH, 'song_meta.json'), encoding="utf-8") as f:
            self.song_meta = pd.DataFrame(json.load(f))

    def intersect_cnt(self, tags, cand_tags, title_tags, genre_tags):

        tags_o = len(list(set(tags) & set(cand_tags))) * 3
        tags_t = len(list(set(tags) & set(title_tags))) * 2
        tags_g = len(list(set(tags) & set(genre_tags)))

        score = tags_o + tags_t + tags_g

        return score

    def mk_rec(self, x):
        t = self.train.copy()
        t['cnt'] = 0

        tags_input = x.tags.copy()
        tags_genre = x.genre_tags.copy()
        tags_title = x.title_tags.copy()

        tag_score = t['tags'].apply(lambda x: self.intersect_cnt(x, tags_input, tags_title, tags_genre))
        t['cnt'] += tag_score

        t = t.sort_values(by='cnt', ascending=False)

        max_cnt = t.cnt.values[0]

        tag_result = []
        song_result = []

        while max_cnt > 0:

            tl = list(t[t['cnt'] == max_cnt]['tags'])
            sl = list(t[t['cnt'] == max_cnt]['songs'])

            tc = Counter([item for sublist in tl for item in sublist]).most_common()
            sc = Counter([item for sublist in sl for item in sublist]).most_common()

            # before updt_date인지 체크
            cand_song = list(map(lambda x: x[0], sc))
            song_result += ArenaUtil.before_updt_date(cand_song, x.updt_date, self.song_meta)

            for i in tc:
                if (i[0] not in x['tags']) & (i[0] not in tag_result):
                    tag_result.append(i[0])

            if ((len(song_result) >= 10) & (len(tag_result) >= 10)):
                break

            max_cnt -= 1

        return [tag_result[:10], song_result[:10 - len(x.songs)]]

    def run(self):
        tqdm.pandas()

        # song 개수가 3개 미만일 경우만 rec_songs, rec_tags 뽑아내기
        # 최종 제출할 때는 val -> test로

        val_not = self.test[(self.test.songs.str.len() >= 3)].copy()
        val_min = self.test[(self.test.songs.str.len() < 3)].copy()
        val_min['rec'] = val_min.progress_apply(lambda x: self.mk_rec(x), axis=1)
        val_min['rec_tags'] = val_min.apply(lambda x: x.rec[0], axis=1)
        val_min['rec_songs'] = val_min.apply(lambda x: x.rec[1], axis=1)
        val_min = val_min.drop(columns='rec')

        self.val = pd.concat([val_not, val_min]).sort_index(ascending=True)

        self.res = []
        for pid in base.val.index:
            self.res.append({
                "id": self.val.loc[pid, "id"],
                "songs": self.val.loc[pid, "rec_songs"],
                "tags": self.val.loc[pid, "rec_tags"]
            })

        ArenaUtil.write_json(self.res, self.FILE_PATH, "res.json")