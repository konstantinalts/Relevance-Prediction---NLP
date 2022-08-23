import os
import numpy as np
import pandas as pd

from tabulate import tabulate
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import DataLoader
from data_processor import DataProcessor


INPUT_DIR = './data/'

# If merged pickle file exists, load it
if os.path.exists(f'{INPUT_DIR}merged.pkl'):
    attr_df = pd.read_pickle(f'{INPUT_DIR}attr.pkl')
    merged = pd.read_pickle(f'{INPUT_DIR}merged.pkl')
# Else, load and proceess data
else:
    # Instantiate DataLoader
    dl = DataLoader(INPUT_DIR)

    # Load datasets into pandas dataframes
    train_df, pro_desc_df, attr_df = dl.load_data()

    # Keep only Brand Name from attributes
    brand_df = attr_df[attr_df.name == "MFG Brand Name"][[
        "product_uid", "value"]].rename(columns={"value": "brand"})

    # Merge the 3 dataframes into one
    merged = pd.merge(train_df, pro_desc_df, how='left', on='product_uid')
    merged = pd.merge(merged, brand_df, how='left', on='product_uid')

    # Instantiate DataProcessor
    dp = DataProcessor()

    # Apply text processing for specific columns
    columns_to_process = ['product_title', 'search_term', 'product_description', 'brand']
    for col in columns_to_process:
        merged[col] = merged[col].apply(dp.str_process)

    # Sabe merged dataframe to a picke file
    merged.to_pickle(f'{INPUT_DIR}merged.pkl')

# If word2vec models exists, load it
if os.path.exists('word2vec.model'):
    w2v_model = Word2Vec.load('word2vec.model')
else:

    combined_training = pd.concat(
        [merged.product_title, merged.product_description, merged.search_term, merged.brand])

    train_data = []
    for i in combined_training:
        train_data.append(i.split())

    w2v_model = Word2Vec(train_data, vector_size=300, min_count=2, window=5, sg=1, workers=4)
    w2v_model.save('word2vec.model')


# Function returning vector representation of a document
def get_embedding_w2v(doc_tokens):
    embeddings = []
    if len(doc_tokens) < 1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv:
                embeddings.append(w2v_model.wv.get_vector(tok, norm=True))
            else:
                embeddings.append(np.random.rand(300))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)


if os.path.exists(f'{INPUT_DIR}merged_w2v.pkl'):
    merged = pd.read_pickle(f'{INPUT_DIR}merged_w2v.pkl')
else:
    # Combile title, description and brand into a single string
    merged['combined'] = merged['product_title'] + merged['product_description'] + merged['brand']

    merged['search_vector'] = merged['search_term'].apply(lambda x: get_embedding_w2v(x.split()))

    merged['title_vector'] = merged['product_title'].apply(lambda x: get_embedding_w2v(x.split()))

    merged['description_vector'] = merged['product_description'].apply(
        lambda x: get_embedding_w2v(x.split()))

    merged['brand_vector'] = merged['brand'].apply(
        lambda x: get_embedding_w2v(x.split()))

    merged['combined_vector'] = merged['combined'].apply(
        lambda x: get_embedding_w2v(x.split()))

    sim_term_title = []
    sim_term_desc = []
    sim_term_brand = []
    sim_term_combined = []

    # Iterate over all rows and compute the normalized full and partio similarity ratios
    for i, row in merged.iloc[:].iterrows():

        sim_term_title.append(
            cosine_similarity(row.search_vector.reshape(1, -1),
                              row.title_vector.reshape(1, -1))[0][0])
        sim_term_desc.append(
            cosine_similarity(row.search_vector.reshape(1, -1),
                              row.description_vector.reshape(1, -1))[0][0])
        sim_term_brand.append(
            cosine_similarity(row.search_vector.reshape(1, -1),
                              row.brand_vector.reshape(1, -1))[0][0])
        sim_term_combined.append(
            cosine_similarity(row.search_vector.reshape(1, -1),
                              row.combined_vector.reshape(1, -1))[0][0])

    merged['sim_term_title'] = pd.Series(sim_term_title)
    merged['sim_term_desc'] = pd.Series(sim_term_desc)
    merged['sim_term_brand'] = pd.Series(sim_term_brand)
    merged['sim_term_combined'] = pd.Series(sim_term_combined)

    # Save to pickle
    merged.to_pickle(f'{INPUT_DIR}merged_w2v.pkl')


# similarity = w2v_model.wv.wmdistance(merged.iloc[0].search_term, merged.iloc[0].combined, norm=True)
# print(f"{similarity:.4f}")
#
# similarity = cosine_similarity(np.array(merged.iloc[0].search_vector).reshape(
#     1, -1), np.array(merged.iloc[0].description_vector).reshape(1, -1))
# # print(f"{similarity:.4f}")
# print(similarity)


# Drop text unnecessary columns
merged.drop(['id', 'product_uid', 'product_title', 'search_term',
             'product_description', 'brand', 'combined', 'search_vector', 'title_vector', 'combined_vector', 'description_vector', 'brand_vector'], axis=1, inplace=True)


# Print
print(merged.iloc[0].sim_term_title)
# print(merged.columns)
# print(tabulate(merged[:5], headers='keys', tablefmt='github'))

#
# Spit dataset into train and test subsets
from sklearn.model_selection import train_test_split
X = merged.loc[:, merged.columns != 'relevance']
y = merged.loc[:, merged.columns == 'relevance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=0)
rf.fit(X_train, y_train.values.ravel())
y_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_pred, y_test)
rf_rmse = np.sqrt(rf_mse)
print('RandomForest RMSE: %.4f' % rf_rmse)

# Ridge Regression
from sklearn.linear_model import Ridge

rg = Ridge(alpha=.1)
rg.fit(X_train, y_train.values.ravel())
y_pred = rg.predict(X_test)
rg_mse = mean_squared_error(y_pred, y_test)
rg_rmse = np.sqrt(rg_mse)
print('Ridge RMSE: %.4f' % rg_rmse)


# Gradient Boosting for Regression
from sklearn.ensemble import GradientBoostingRegressor

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1,
                                random_state=0, loss='ls').fit(X_train, y_train.values.ravel())
y_pred = est.predict(X_test)
est_mse = mean_squared_error(y_pred, y_test)
est_rmse = np.sqrt(est_mse)
print('Gradient boosting RMSE: %.4f' % est_rmse)

# XG Boost
import xgboost
from sklearn.metrics import mean_squared_error
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0,
                           subsample=0.75, colsample_bytree=1, max_depth=7)
xgb.fit(X_train, y_train.values)
y_pred = xgb.predict(X_test)
xgb_mse = mean_squared_error(y_pred, y_test)
xgb_rmse = np.sqrt(xgb_mse)
print('Xgboost RMSE: %.4f' % xgb_rmse)
