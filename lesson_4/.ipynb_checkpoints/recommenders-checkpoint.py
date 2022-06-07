import pandas as pd
import numpy as np
# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

class MainRecommender:
    
    def __init__(self, data, weighting=False):
        self.top_n = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index().sort_values('quantity', ascending=False)        
        self.user_item_matrix = self.prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dict(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
    
    @staticmethod
    def prepare_matrix(data):
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', 
                                  aggfunc='count', 
                                  fill_value=0)
        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dict(user_item_matrix):
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values
        
        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))
        
        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))
        
        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model
    
    def get_rec_similar_items(self, item_id, N=2):
        """Рекомендуем N товаров, похожие на товар item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=N)
        top_rec=recs[1][0]
        return top_rec
    
    def get_similar_items_recommendation(self, user, N=5):
        
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        top = self.top_n[self.top_n['user_id']==user].head(N)
        recs = top['item_id'].apply(lambda x: self.get_rec_similar_items(x)).tolist()
        return recs

        
       

