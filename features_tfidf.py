import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import DataLoader
from data_processor import DataProcessor

# Used to count how much time it takes to run the script
import time
start = time.time()

INPUT_DIR = './data/'


# If tfidf data exist, load them
if os.path.exists(f'{INPUT_DIR}merged_tfidf.pkl'):
    merged = pd.read_pickle(f'{INPUT_DIR}merged_tfidf.pkl')
else:

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

        # Save merged dataframe to a picke file
        merged.to_pickle(f'{INPUT_DIR}merged.pkl')

    # TF-IDF and Truncated SVD initialization (to extract concept using Latent semantics analysis(LSA))
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')  # stop words
    # n_components=100 to extract concepts using LSA
    svd = TruncatedSVD(n_components=100, random_state=2019)

    # Creating pipeline to execute TF-IDF and SVD in one step
    pipe = Pipeline(steps=[('tfidf', tfidf), ('svd', svd)])
    # pipe = Pipeline(steps=[('tfidf', tfidf)])

    merged["prod_desc_merge"] = merged["product_description"].map(
        str) + ' ' + merged["brand"].fillna('').map(str)

    # Perform fit and transform function of pipeline to convert text(in each feature) into vectors and reducing them
    merged["product_title"] = pipe.fit_transform(merged["product_title"])
    merged["search_term"] = pipe.fit_transform(merged["search_term"])
    merged["prod_desc_merge"] = pipe.fit_transform(merged["prod_desc_merge"])

    merged.to_pickle(f'{INPUT_DIR}merged_tfidf.pkl')


# Drop tring values that are not needed for the model
merged.drop(['product_uid', "product_description", "brand"], axis=1, inplace=True)

print(merged.head(5))

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
xgb.fit(X_train, y_train.values.ravel())
y_pred = xgb.predict(X_test)
xgb_mse = mean_squared_error(y_pred, y_test)
xgb_rmse = np.sqrt(xgb_mse)
print('Xgboost RMSE: %.4f' % xgb_rmse)
