import pandas as pd
from catboost import CatBoostRanker
import numpy as np

test_df = pd.read_csv('test_df.csv')

queries_test = test_df['search_id'].values
y_test = test_df['target'].values
X_test = test_df.drop(columns=['search_id', 'target']).values

model = CatBoostRanker()
model.load_model('/models/catboost_model.cbm')

ndcg_test = model.score(X_test, y_test, group_id=queries_test, top=len(y_test))
print(f"NDCG Score on Test Data: {ndcg_test}")