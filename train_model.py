import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np

train_df = pd.read_csv('train_df.csv')

queries = train_df['search_id'].unique()
queries_train, queries_val = train_test_split(queries, test_size=0.2, random_state=42)

X_train = train_df[(train_df['search_id'].isin(queries_train))]
X_val = train_df[(train_df['search_id'].isin(queries_val))]
print(f"Количество строк в train датасете:\t {len(X_train)}")
print(f"Количество строк в val датасете:\t {len(X_val)}")

queries_train = X_train['search_id'].values
y_train = X_train['target'].values
X_train = X_train.drop(columns=['search_id', 'target']).values

queries_val = X_val['search_id'].values
y_val = X_val['target'].values
X_val = X_val.drop(columns=['search_id', 'target']).values

print(f"Количество групп в train датасете:\t {len(np.unique(queries_train))}")
print(f"Количество групп в val датасете:\t {len(np.unique(queries_val))}")

train = Pool(
    data=X_train,
    label=y_train,
    group_id=queries_train
)

val = Pool(
    data=X_val,
    label=y_val,
    group_id=queries_val
)

default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0,
    'early_stopping_rounds' : 100,
}

def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=val):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    
    if additional_params is not None:
        parameters.update(additional_params)
        
    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, verbose=5)
   
    return model

model = fit_model('QueryRMSE')

ndcg_train = model.score(X_train, y_train, group_id=queries_train, top=len(y_train))
print(f"NDCG Score on Train Data: {ndcg_train}")
ndcg_val = model.score(X_val, y_val, group_id=queries_val, top=len(y_val))
print(f"NDCG Score on Val Data: {ndcg_val}")

model.save_model('/models/catboost_model.cbm')