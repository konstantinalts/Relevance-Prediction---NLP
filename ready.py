# code from https://towardsdatascience.com/predict-search-relevance-using-machine-learning-for-online-retailers-5d3e47acaa33

import numpy as np
import pandas as pd
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
import random
random.seed(2018)

import os

INPUT_DIR = './data/'


# Load 3 datasets into Dataframe and return them
def create_df(input_dir=INPUT_DIR):
    train_df = pd.read_csv(f'{INPUT_DIR}train.csv', encoding="ISO-8859-1")
    pro_desc_df = pd.read_csv(f'{INPUT_DIR}product_descriptions.csv')
    attr_df = pd.read_csv(f'{INPUT_DIR}attributes.csv')

    return train_df, pro_desc_df, attr_df


# Create dataframes, store them to pickle files and return them
def create_pickle(input_dir=INPUT_DIR):
    train_df, pro_desc_df, attr_df = create_df()
    train_df.to_pickle(f'{INPUT_DIR}train.pkl')
    pro_desc_df.to_pickle(f'{INPUT_DIR}pro_desc.pkl')
    attr_df.to_pickle(f'{INPUT_DIR}attr.pkl')

    return train_df, pro_desc_df, attr_df


# If pickle file exist, return those.
# Else, creath them
def load_data(input_dir=INPUT_DIR):
    if os.path.exists(f'{INPUT_DIR}train.pkl') and os.path.exists(f'{INPUT_DIR}pro_desc.pkl') and os.path.exists(f'{INPUT_DIR}attr.pkl'):
        return pd.read_pickle(f'{INPUT_DIR}train.pkl'), pd.read_pickle(f'{INPUT_DIR}pro_desc.pkl'), pd.read_pickle(f'{INPUT_DIR}attr.pkl')
    else:
        return create_pickle()


# Load data
df_train, df_pro_desc, df_attr = load_data()

df_brand = df_attr[df_attr.name == "MFG Brand Name"][[
    "product_uid", "value"]].rename(columns={"value": "brand"})
df_all = pd.merge(df_train, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')


# df_all['search_term'] = df_all['search_term'].map(
#     lambda x: google_dict[x] if x in google_dict.keys() else x)


def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. ", s)
        s = re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. ", s)
        s = re.sub(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. ", s)
        s = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. ", s)
        s = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr ", s)
        s = re.sub(
            r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. ", s)
        s = re.sub(
            r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr ", s)
        # Deal with special characters
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("&nbsp;", " ")
        s = s.replace("&amp;", "&")
        s = s.replace("&#39;", "'")
        s = s.replace("/>/Agt/>", "")
        s = s.replace("</a<gt/", "")
        s = s.replace("gt/>", "")
        s = s.replace("/>", "")
        s = s.replace("<br", "")
        s = s.replace("<.+?>", "")
        s = s.replace("[ &<>)(_,;:!?\+^~@#\$]+", " ")
        s = s.replace("'s\\b", "")
        s = s.replace("[']+", "")
        s = s.replace("[\"]+", "")
        s = s.replace("-", " ")
        s = s.replace("+", " ")
        # Remove text between paranthesis/brackets)
        s = s.replace("[ ]?[[(].+?[])]", "")
        # remove sizes
        s = s.replace("size: .+$", "")
        s = s.replace("size [0-9]+[.]?[0-9]+\\b", "")

        return " ".join([stemmer.stem(re.sub('[^A-Za-z0-9-./]', ' ', word)) for word in s.lower().split()])
    else:
        return "null"


df_all['product_title'] = df_all['product_title'].apply(str_stem)
df_all['search_term'] = df_all['search_term'].apply(str_stem)
df_all['product_description'] = df_all['product_description'].apply(str_stem)
df_all['brand'] = df_all['brand'].apply(str_stem)

a = 0
for i in range(a, a + 2):
    print(df_all.product_title[i])
    print(df_all.search_term[i])
    print(df_all.product_description[i])
    print(df_all.brand[i])
    print(df_all.relevance[i])
    print()


def str_common_word(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    words, count = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            count += 1
    return count


def str_whole_word(str1, str2, i_):
    str1, str2 = str1.lower().strip(), str2.lower().strip()
    count = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return count
        else:
            count += 1
            i_ += len(str1)
    return count


df_all['word_len_of_search_term'] = df_all['search_term'].apply(
    lambda x: len(x.split())).astype(np.int64)
df_all['word_len_of_title'] = df_all['product_title'].apply(
    lambda x: len(x.split())).astype(np.int64)
df_all['word_len_of_description'] = df_all['product_description'].apply(
    lambda x: len(x.split())).astype(np.int64)
df_all['word_len_of_brand'] = df_all['brand'].apply(lambda x: len(x.split())).astype(np.int64)
# Create a new column that combine "search_term", "product_title" and "product_description"
df_all['product_info'] = df_all['search_term'] + "\t" + \
    df_all['product_title'] + "\t" + df_all['product_description']
# Number of times the entire search term appears in product title.
df_all['query_in_title'] = df_all['product_info'].map(
    lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
# Number of times the entire search term appears in product description
df_all['query_in_description'] = df_all['product_info'].map(
    lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
# Number of words that appear in search term also appear in product title.
df_all['word_in_title'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
# Number of words that appear in search term also appear in production description.
df_all['word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
# The ratio of product title word length to search term word length
df_all['query_title_len_prop'] = df_all['word_len_of_title'] / df_all['word_len_of_search_term']
# The ratio of product description word length to search term word length
df_all['query_desc_len_prop'] = df_all['word_len_of_description'] / df_all['word_len_of_search_term']
# The ratio of product title and search term common word count to search term word count
df_all['ratio_title'] = df_all['word_in_title'] / df_all['word_len_of_search_term']
# The ratio of product description and search term common word cout to search term word count.
df_all['ratio_description'] = df_all['word_in_description'] / df_all['word_len_of_search_term']
# new column that combine "search_term", "brand" and "product_title".
df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand'] + "\t" + df_all['product_title']
# Number of words that appear in search term also apprears in brand.
df_all['word_in_brand'] = df_all['attr'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
# The ratio of search term and brand common word count to brand word count
df_all['ratio_brand'] = df_all['word_in_brand'] / df_all['word_len_of_brand']

df_all.drop(['id', 'product_uid', 'product_title', 'search_term',
             'product_description', 'brand', 'product_info', 'attr'], axis=1, inplace=True)

df_all.columns

from sklearn.model_selection import train_test_split
X = df_all.loc[:, df_all.columns != 'relevance']
y = df_all.loc[:, df_all.columns == 'relevance']
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
