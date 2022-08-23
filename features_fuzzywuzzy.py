import os
import numpy as np
import pandas as pd

from tabulate import tabulate
from fuzzywuzzy import fuzz, process

from data_loader import DataLoader
from data_processor import DataProcessor


INPUT_DIR = './data/'

# Instantiate DataLoader
dl = DataLoader(INPUT_DIR)

if dl.pickle_exists('merged_fuzzy'):
    merged = dl.load_pickle('merged_fuzzy')
else:
    # If merged pickle file exists, load it
    if dl.pickle_exists('merged_processed'):
        merged = load_pickle('merged_processed')
        attr_df = load_pickle('attr')

    # Else, load and proceess data
    else:
        # Load original merged
        merged = dl.load_merged()

        # Instantiate DataProcessor
        dp = DataProcessor()

        # Apply text processing for specific columns
        columns_to_process = ['product_title', 'search_term', 'product_description', 'brand']
        for col in columns_to_process:
            merged[col] = merged[col].apply(dp.str_process)

        # Save merged dataframe to a picke file
        dl.save_pickle(merged, 'merged_processed')

    # Create empty list for store similarity scores
    sim_term_title_full = []
    sim_term_title_partial = []
    sim_term_desc_full = []
    sim_term_desc_partial = []
    sim_term_brand_full = []
    sim_term_brand_partial = []

    # Iterate over all rows and compute the normalized full and partio similarity ratios
    for i, row in merged.iloc[:].iterrows():

        sim_term_title_full.append(fuzz.ratio(row.search_term, row.product_title) / 100)
        sim_term_title_partial.append(fuzz.partial_ratio(row.search_term, row.product_title) / 100)
        sim_term_desc_full.append(fuzz.ratio(row.search_term, row.product_description) / 100)
        sim_term_desc_partial.append(fuzz.partial_ratio(
            row.search_term, row.product_description) / 100)
        sim_term_brand_full.append(fuzz.ratio(row.search_term, row.brand) / 100)
        sim_term_brand_partial.append(fuzz.partial_ratio(row.search_term, row.brand) / 100)

    # Add similarity measures to df
    merged['sim_term_title_full'] = pd.Series(sim_term_title_full)
    merged['sim_term_title_partial'] = pd.Series(sim_term_title_partial)
    merged['sim_term_desc_full'] = pd.Series(sim_term_desc_full)
    merged['sim_term_desc_partial'] = pd.Series(sim_term_desc_partial)
    merged['sim_term_brand_full'] = pd.Series(sim_term_brand_full)
    merged['sim_term_brand_partial'] = pd.Series(sim_term_brand_partial)

    # Save to pickles
    dl.save_pickle(merged, 'merged_fuzzy')

# Drop text unnecessary columns
merged.drop(['id', 'product_uid', 'product_title', 'search_term',
             'product_description', 'brand'], axis=1, inplace=True)


# Print
print(tabulate(merged[:5], headers='keys', tablefmt='github'))

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
