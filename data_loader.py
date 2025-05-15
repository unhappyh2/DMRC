import pandas as pd
import os
import numpy as np
import torch

dataset = {"movie":'./data/ml-latest-small/',
           "music":"./data/music/hetrec2011-lastfm-2k/",
           "book":"./data/book/",
           "movie2":"./data/ml-1m/",
           "anime":"./data/anime/"}

class loader():
    def __init__(self,database):
        self.database = database
        self.data_mapping = {"movie":['ratings.csv','movies.csv'],
                             "music":["user_artists.dat","artists.dat"],
                             "book":["Ratings.csv","Books.csv"],
                             "movie2":['ratings.dat','movies.dat'],
                             "anime":["rating.csv","anime.csv"]}

    def read_data(self,database,rating_threshold = 4):
        num_genres = 0
        genre_edge_index = None
        if database == "movie" :
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], index_col='userId')
            user_mapping = {index: i for i, index in enumerate(df.index.unique())}
            df = pd.read_csv(dataset[database]+self.data_mapping[database][1], index_col='movieId')
            item_mapping = {index: i for i, index in enumerate(df.index.unique())}

            # 读取电影信息文件构造 item_id -> genres 映射
            movie_info_df = pd.read_csv(dataset[database]+self.data_mapping[database][1])
            genre_set = set()
            movie_genres = {}  # item_id -> [genre]
            for row in movie_info_df.itertuples():
                movie_id = row.movieId
                if movie_id in item_mapping:
                    genres = str(row.genres).split('|')
                    mapped_id = item_mapping[movie_id]
                    movie_genres[mapped_id] = genres
                    genre_set.update(genres)

            # 构建genre到索引的映射
            genre2id = {genre: i for i, genre in enumerate(sorted(genre_set))}
            # print("genre2id:", genre2id)
            #print("item_mapping:", item_mapping)
            # 构建 movie -> genre 的边（可用于知识图谱图建模）
            genre_edge_index = [[], []]
            for movie_id, genres in movie_genres.items():
                for genre in genres:
                    genre_id = genre2id[genre]
                    genre_edge_index[0].append(movie_id)
                    genre_edge_index[1].append(genre_id)

            genre_edge_index = torch.tensor(genre_edge_index)
            #print("genre_edge_index", genre_edge_index[:,:6])
            num_genres = len(genre_set)
            # 获取用户和电影数量
            num_users = len(user_mapping)
            num_items = len(item_mapping)
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0])
            edge_index = None
            src = [user_mapping[index] for index in df['userId']]
            dst = [item_mapping[index] for index in df['movieId']]
            edge_attr = torch.from_numpy(df['rating'].values).view(-1, 1).to(torch.long) >= rating_threshold  # 将数组转化为tensor张量
            edge_index = [[], []]
            for i in range(edge_attr.shape[0]):
                if edge_attr[i]:
                    edge_index[0].append(src[i])
                    edge_index[1].append(dst[i])

            edge_index = torch.tensor(edge_index)
            edge_arr = np.array(df['rating'].values)

        elif database == "movie2":
            # 读取数据
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], index_col=0, sep="::", header=None, engine="python",encoding="latin1")
            user_mapping = {index: i for i, index in enumerate(df.index.unique())}
            df = pd.read_csv(dataset[database]+self.data_mapping[database][1], index_col=0, sep="::", header=None, engine="python",encoding="latin1")
            item_mapping = {index: i for i, index in enumerate(df.index.unique())}

            # 读取电影信息文件构造 item_id -> genres 映射
            movie_info_df = pd.read_csv(dataset[database]+self.data_mapping[database][1], sep="::", header=None, engine="python",encoding="latin1")
            movie_info_df.columns = ['movieId', 'title', 'genres']
            genre_set = set()
            movie_genres = {}  # item_id -> [genre]
            for row in movie_info_df.itertuples():
                movie_id = row.movieId
                if movie_id in item_mapping:
                    genres = str(row.genres).split('::')
                    mapped_id = item_mapping[movie_id]
                    movie_genres[mapped_id] = genres
                    genre_set.update(genres)

            # 构建genre到索引的映射
            genre2id = {genre: i for i, genre in enumerate(sorted(genre_set))}
            # print("genre2id:", genre2id)
            #print("item_mapping:", item_mapping)
            # 构建 movie -> genre 的边（可用于知识图谱图建模）
            genre_edge_index = [[], []]
            for movie_id, genres in movie_genres.items():
                for genre in genres:
                    genre_id = genre2id[genre]
                    genre_edge_index[0].append(movie_id)
                    genre_edge_index[1].append(genre_id)

            genre_edge_index = torch.tensor(genre_edge_index)
            #print("genre_edge_index", genre_edge_index[:,:6])
            num_genres = len(genre_set)

            # 获取用户和电影数量
            num_users = len(user_mapping)
            num_items = len(item_mapping)
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], sep="::", header=None, engine="python",encoding="latin1")
            edge_index = None
            src = [user_mapping[index] for index in df[0]]
            dst = [item_mapping[index] for index in df[1]]
            edge_attr = torch.from_numpy(df[3].values).view(-1, 1).to(torch.long) >= rating_threshold  # 将数组转化为tensor张量
            edge_index = [[], []]
            for i in range(edge_attr.shape[0]):
                if edge_attr[i]:
                    edge_index[0].append(src[i])
                    edge_index[1].append(dst[i])

            edge_index = torch.tensor(edge_index)
            edge_arr = np.array(df[3].values)


        elif database == "music":
            
            # 读取数据
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], index_col='userID', sep="\t")
            user_mapping = {index: i for i, index in enumerate(df.index.unique())}
            df = pd.read_csv(dataset[database]+self.data_mapping[database][1], index_col='id', sep="\t")
            item_mapping = {index: i for i, index in enumerate(df.index.unique())}
            # 获取用户和电影数量
            num_users = len(user_mapping)
            num_items = len(item_mapping)
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], sep="\t")
            edge_index = None
            src = [user_mapping[index] for index in df['userID']]
            dst = [item_mapping[index] for index in df['artistID']]
            edge_attr = torch.from_numpy(df["weight"].values).view(-1, 1).to(torch.long) >= rating_threshold  # 将数组转化为tensor张量
            edge_index = [[], []]
            for i in range(edge_attr.shape[0]):
                if edge_attr[i]:
                    edge_index[0].append(src[i])
                    edge_index[1].append(dst[i])

            edge_index = torch.tensor(edge_index)
            edge_arr = np.array(df["weight"].values)


        elif database == "book":

            # 读取数据
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], index_col='User-ID', sep=None, engine="python", encoding="utf-8")
            user_mapping = {index: i for i, index in enumerate(df.index.unique())}
            df = pd.read_csv(dataset[database]+self.data_mapping[database][1], index_col='ISBN', sep=None, engine="python", encoding="utf-8")
            item_mapping = {index: i for i, index in enumerate(df.index.unique())}
            # 
            num_users = len(user_mapping)
            num_items = len(item_mapping)
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], sep=None)
            print(df.head)
            edge_index = None
            unmapped = set(df['ISBN']) - set(item_mapping.keys())
            print("ISBN:", len(unmapped))
            src = [user_mapping[index] for index in df['User-ID']]
            dst = [item_mapping[index] for index in df['ISBN']]
            edge_attr = torch.from_numpy(df['Rating'].values).view(-1, 1).to(torch.long) >= rating_threshold  # 将数组转化为tensor张量
            edge_index = [[], []]
            for i in range(edge_attr.shape[0]):
                if edge_attr[i]:
                    edge_index[0].append(src[i])
                    edge_index[1].append(dst[i])

            edge_index = torch.tensor(edge_index)
            edge_arr = np.array(df['Rating'].values)
        
        elif database == "anime":
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], index_col='user_id', sep=None, engine="python", encoding="utf-8")
            user_mapping = {index: i for i, index in enumerate(df.index.unique())}
            df = pd.read_csv(dataset[database]+self.data_mapping[database][1], index_col='anime_id', sep=None, engine="python", encoding="utf-8")
            item_mapping = {index: i for i, index in enumerate(df.index.unique())}

            num_users = len(user_mapping)
            num_items = len(item_mapping)
            df = pd.read_csv(dataset[database]+self.data_mapping[database][0], sep=None)
            print(df.head)
            df = df[df['anime_id'].isin(item_mapping)]
            edge_index = None
            unmapped = set(df['anime_id']) - set(item_mapping.keys())
            print("未映射的anime_id:", unmapped)
            src = [user_mapping[index] for index in df['user_id']]
            dst = [item_mapping[index] for index in df['anime_id']]
            edge_attr = torch.from_numpy(df['rating'].values).view(-1, 1).to(torch.long) >= 2*rating_threshold  # 将数组转化为tensor张量
            edge_index = [[], []]
            for i in range(edge_attr.shape[0]):
                if edge_attr[i]:
                    edge_index[0].append(src[i])
                    edge_index[1].append(dst[i])

            edge_index = torch.tensor(edge_index)
            edge_arr = np.array(df['rating'].values)
            
        return edge_index,genre_edge_index, edge_arr, num_users, num_items,num_genres, user_mapping, item_mapping
        

if __name__ =="__main__":
    database ="movie2"
    loaders = loader(database)
    edge_index, genre_index, edge_arr, num_users, num_items,num_genres, user_mapping, movie_mapping =loaders.read_data(database)
    print("edge_index.shape:",edge_index.shape)
    print("edge_arr.shape:",edge_arr.shape)
    print("num_genres:",num_genres)
    print("num_users:",num_users)
    print("num_items:",num_items)