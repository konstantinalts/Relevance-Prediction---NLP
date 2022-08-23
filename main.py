import os
import numpy as np
import pandas as pd

from time import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tabulate import tabulate

from data_loader import DataLoader
from data_processor import DataProcessor
from data_transfomer import DataTransformer

INPUT_DIR = './data/'

# Instantiate DataLoader
dl = DataLoader(INPUT_DIR)
merged = dl.load_merged()

# Load the pre-proceessed data
if dl.pickle_exists('merged_processed'):
    merged = dl.load_pickle('merged_processed')

# Pre-proceess the data and save them into a picke
else:
    # Instantiate DataProcessor
    dp = DataProcessor()

    # Apply text processing for specific columns
    columns_to_process = ['product_title', 'search_term', 'product_description', 'brand']
    for col in columns_to_process:
        merged[col] = merged[col].apply(dp.str_process)

    # Save merged dataframe to a picke file
    dl.save_pickle(merged, 'merged_processed')

dt = DataTransformer(dl, merged)

dfs = []

start = time()
df_lev = dt.fuzzy()
dfs.append((df_lev, 'Levenshtein Distance', f'{time() - start:.4f}'))

start = time()
df_tfidf = dt.tfidf()
dfs.append((df_tfidf, 'TF-IDF', f'{time() - start: .4f}'))

start = time()
df_w2c_1_300_2_5 = dt.word2vec(sg=1, vector_size=300, min_count=2, window=5, workers=4)
dfs.append((df_w2c_1_300_2_5, 'Word2Vec_1-300-2-5', f'{time() - start: .4f}'))

start = time()
df_w2c_0_300_2_5 = dt.word2vec(sg=0, vector_size=300, min_count=2, window=5, workers=4)
dfs.append((df_w2c_0_300_2_5, 'Word2Vec_0-300-2-5', f'{time() - start: .4f}'))

start = time()
df_w2c_1_100_2_5 = dt.word2vec(sg=1, vector_size=100, min_count=2, window=5, workers=4)
dfs.append((df_w2c_1_100_2_5, 'Word2Vec_1-100-2-5', f'{time() - start: .4f}'))

start = time()
df_w2c_0_100_2_5 = dt.word2vec(sg=0, vector_size=100, min_count=2, window=5, workers=4)
dfs.append((df_w2c_0_100_2_5, 'Word2Vec_0-100-2-5', f'{time() - start: .4f}'))

start = time()
df_d2c_0_300_2_5 = dt.doc2vec(dm=0, vector_size=300, min_count=2, window=5, workers=4)
dfs.append((df_d2c_0_300_2_5, 'Doc2Vec_0-300-2-5', f'{time() - start: .4f}'))

start = time()
df_d2c_1_300_2_5 = dt.doc2vec(dm=1, vector_size=300, min_count=2, window=5, workers=4)
dfs.append((df_d2c_1_300_2_5, 'Doc2Vec_1-300-2-5', f'{time() - start: .4f}'))

# Create a dictionany for different classifiers, along with their names and initialization arguments
models = []
models.append({'name': 'RandomForest', 'model': RandomForestRegressor(), 'args': {
    'n_estimators': 100, 'max_depth': 6, 'random_state': 0}})
models.append({'name': 'Ridge', 'model': Ridge(),
               'args': {'alpha': .1}})
models.append({'name': 'Gradient boosting', 'model': GradientBoostingRegressor(),
               'args': {'n_estimators': 100, 'learning_rate': .1, 'max_depth': 1, 'random_state': 0, 'loss': 'ls'}})
models.append({'name': 'XGboost', 'model': XGBRegressor(), 'args': {
    'n_estimators': 100, 'learning_rate': .08, 'max_depth': 7, 'gamma': 0, 'subsample': .75, 'colsample_bytree': 1}})

results = []

# Iterate over all the data frames
for df in dfs:
    print(f'Start training {df[1]}.')
    start = time()
    result = {}
    result['Name'] = df[1]
    result['Build time (s)'] = df[2]

    # Spit dataset into train and test subsets
    X = df[0].loc[:, df[0].columns != 'relevance']
    y = df[0].loc[:, df[0].columns == 'relevance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Build and train models and evaluate their performance
    for model in models:
        # Pass arguments to __init__ before initializing a classifier instance
        m = model['model'].__init__(**model['args'])
        # Build model
        m = model['model']
        # Train model and make predictions
        m.fit(X_train, y_train.values.ravel())
        y_pred = m.predict(X_test)
        # Calculate and store the Root Mean Square Error
        rmse = np.sqrt(mean_squared_error(y_pred, y_test))
        result[model['name']] = f'{rmse:.4f}'

    result['Training time (s)'] = f'{time() - start:.4f}'

    results.append(result)


df = pd.DataFrame(results, columns=['Name', 'RandomForest', 'Ridge',
                                    'Gradient boosting', 'XGboost', 'Build time (s)', 'Training time (s)'])
print('Final results:')
print(tabulate(df, headers='keys', tablefmt='github'))
