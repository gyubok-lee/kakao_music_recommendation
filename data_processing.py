'''
song_meta에 플레이리스트 id 리스트 추가.
장르 결측치 처리.
'''

# 플레이리스트 칼럼 추가
t_explode_bysongs = train.explode('songs')

blank = []
for i in range(len(song_meta)):
    blank.append([])
song_playlist_dict = dict(zip(song_meta.index,blank))

playlist_id = t_explode_bysongs[['id','songs']].reset_index()
playlist_id.head()
for i in playlist_id.index :
    song = playlist_id.loc[i,'songs']
    playlist = playlist_id.loc[i,'id']

    song_playlist_dict[song].append(playlist)

playlist_list = []
for value in song_playlist_dict.values():
    playlist_list.append(value)
song_meta_with_playlist = song_meta.copy()
song_meta_with_playlist['playlist'] = playlist_list
song_meta_with_playlist.to_csv('song_meta_with_playlist.csv',index=False,encoding='utf-8')

# 장르 결측치 채우기
song_meta = pd.read_json('song_meta_with_playlist.json',typ='frame',encoding='utf-8')

genre_gn_all = pd.DataFrame(genre_gn_all,columns=['gnr_name']).reset_index().rename(columns={'index':'gnr_code'})
s_explode_byartist = song_meta.explode('artist_name_basket')
s=s_explode_byartist.copy()

s=s[['artist_name_basket','id','song_gn_gnr_basket']]
artist_sort=s.sort_values(by='artist_name_basket',ascending=True)
artist_sort['artist']=artist_sort['artist_name_basket']
artist_sort['gnr']=artist_sort['song_gn_gnr_basket']
artist_sort=artist_sort[['id','artist','gnr']]
artist_sort_reset=artist_sort.reset_index()
artist_gnr=artist_sort_reset.drop([artist_sort_reset.index[0],artist_sort_reset.index[1],artist_sort_reset.index[2]])

#가수의 별 장르 모음
artist_gnrsum = artist_gnr.groupby('artist')['gnr'].sum()

#가수 장르 모음에서 max 장르 구하기
max_gnr = []
for i in artist_gnrsum :
    gnr_dict = {}
    for g in i :
        if g in gnr_dict:
            gnr_dict[g]+= 1
        else:
            gnr_dict[g]=1
    if len(gnr_dict) == 0 :
        max_gnr.append(-1)
    else:
        key_max = max(gnr_dict.keys(),key=(lambda k:gnr_dict[k]))
        max_gnr.append(key_max)

#가수 중복 제거하고 max 장르 넣어주기
artist_gnr1=artist_gnr.copy()
artist_gnr2=artist_gnr1.drop_duplicates('artist',keep='first')
artist_gnr3=artist_gnr2.copy()
artist_gnr3['max_freq_gnr']=max_gnr
artist_gnr3.head(3)

song_gnrna =song_meta[song_meta.apply(lambda x: len(x['song_gn_gnr_basket']) == 0,axis=1)][['id','artist_name_basket','song_gn_gnr_basket']]

#merge전 전처리
song_gnrna_explode=song_gnrna.explode('artist_name_basket')
before_merge=song_gnrna_explode.copy()
before_merge['artist']=song_gnrna_explode['artist_name_basket']
before_merge['gnr']=song_gnrna_explode['song_gn_gnr_basket']
before_merge2=before_merge[['artist']]
before_merge2.head()

before_merge2['song_id'] = list(before_merge2.index)

#위에 장르 없는 곡에 가수별로 max 장르 넣어주기 성공
fill_gnrna=pd.merge(before_merge2,artist_gnr3,on='artist',how='left')
fill_gnrna['max_freq_gnr'].value_counts()

# t_explode_bysongs에 각 곡의 장르를 찾아 추가하기 (t_explode_bysongs_merge_gnr)
t_explode_bysongs = train.explode('songs')
song_meta_id_gnr = song_meta[['id', 'song_gn_gnr_basket']]

gnr_list = []
for i in t_explode_bysongs['songs']:
    gnr_list.append(song_meta_id_gnr.loc[i, 'song_gn_gnr_basket'])

t_explode_bysongs_merge_gnr = t_explode_bysongs.copy()
t_explode_bysongs_merge_gnr['gnr'] = gnr_list

#플레이리스트 별로 가장 빈도수가 높은 장르 찾아내기 (max_gnr)
playlist_gnrsum = t_explode_bysongs_merge_gnr.groupby('id')['gnr'].sum()
max_gnr = []
for i in playlist_gnrsum :
    gnr_dict = {}
    for g in i :
        if g in gnr_dict:
            gnr_dict[g]+= 1
        else:
            gnr_dict[g]=1
    if len(gnr_dict) == 0 :
        max_gnr.append(-1)
    else:
        key_max = max(gnr_dict.keys(),key=(lambda k:gnr_dict[k]))
        max_gnr.append(key_max)

#train (플레이리스트)에 '가장 빈도수 높은 장르'(most_freq_gnr) 열 추가
train_plylst = train.copy()
train_plylst['most_freq_gnr'] = max_gnr #train_copy라 되어 있는데 일단 수정함.
train_plylst.set_index(train['id'], inplace=True)

fill_gnrna_2 = fill_gnrna.copy()
fill_gnrna_2 = fill_gnrna_2.merge(song_meta[['id','playlist']],on='id')
fill_gnrna_2 = fill_gnrna_2[fill_gnrna_2.apply(lambda x : len(x['gnr'])==0,axis=1)]

fill_gnrna_2.set_index(fill_gnrna_2['song_id'],inplace=True)
fill_gnrna_2 = fill_gnrna_2.drop_duplicates(['song_id'])

playlist_gnr_basket = []

for i in fill_gnrna_2.index :
    gnr_list = []
    for j in fill_gnrna_2.loc[i,'playlist']:
        gnr_list.append(train_plylst.loc[j,'most_freq_gnr'])
    playlist_gnr_basket.append(gnr_list)
fill_gnrna_2['gnr'] = playlist_gnr_basket

print('대체되지 못한 행 개수 : ',+len(fill_gnrna_2[fill_gnrna_2.apply(lambda x : len(x['gnr']) == 0, axis=1)]))

#song_meta원본 데이터에 결합 후 저장
song_meta_final = song_meta.copy()
fill_gnrna.set_index(fill_gnrna['song_id'],inplace=True)

gnr_list= list(song_meta_final['song_gn_gnr_basket'])
for i in fill_gnrna.index:
    gnr_list[i] = fill_gnrna.loc[i,'gnr']
for i in fill_gnrna_2.index :
    gnr_list[i] = fill_gnrna_2.loc[i,'gnr']
song_meta_final['song_gn_gnr_basket'] = gnr_list