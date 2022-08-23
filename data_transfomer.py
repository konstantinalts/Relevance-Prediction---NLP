import os
import time
import numpy as np
import pandas as pd

from fuzzywuzzy import fuzz, process
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

from tqdm import tqdm


class DataTransformer():
    def __init__(self, dl, merged):
        self.dl = dl
        self.merged = merged
        self.model_dir = self.dl.directory('./model/')

    def fuzzy(self):
        if self.dl.pickle_exists('merged_fuzzy'):
            print('Loading merged_fuzzy pickle.')
            merged = self.dl.load_pickle('merged_fuzzy')
        else:
            print('Creating merged_fuzzy dataset.')
            merged = self.merged.copy()
            attr_df = self.dl.load_pickle('attr')

            # Create empty list for store similarity scores
            sim_term_title_full = []
            sim_term_title_partial = []
            sim_term_desc_full = []
            sim_term_desc_partial = []
            sim_term_brand_full = []
            sim_term_brand_partial = []

            # Iterate over all rows and compute the normalized full and partio similarity ratios
            print('  Calculating similarity with Levenshtein distance:')
            for i, row in tqdm(merged.iloc[:].iterrows()):

                sim_term_title_full.append(fuzz.ratio(row.search_term, row.product_title) / 100)
                sim_term_title_partial.append(fuzz.partial_ratio(
                    row.search_term, row.product_title) / 100)
                sim_term_desc_full.append(fuzz.ratio(
                    row.search_term, row.product_description) / 100)
                sim_term_desc_partial.append(fuzz.partial_ratio(
                    row.search_term, row.product_description) / 100)
                sim_term_brand_full.append(fuzz.ratio(row.search_term, row.brand) / 100)
                sim_term_brand_partial.append(
                    fuzz.partial_ratio(row.search_term, row.brand) / 100)

            # Add similarity measures to df
            merged['sim_term_title_full'] = pd.Series(sim_term_title_full)
            merged['sim_term_title_partial'] = pd.Series(sim_term_title_partial)
            merged['sim_term_desc_full'] = pd.Series(sim_term_desc_full)
            merged['sim_term_desc_partial'] = pd.Series(sim_term_desc_partial)
            merged['sim_term_brand_full'] = pd.Series(sim_term_brand_full)
            merged['sim_term_brand_partial'] = pd.Series(sim_term_brand_partial)

            # Save to pickles
            self.dl.save_pickle(merged, 'merged_fuzzy')

        # Drop text unnecessary columns
        merged.drop(['id', 'product_uid', 'product_title', 'search_term',
                     'product_description', 'brand'], axis=1, inplace=True)

        return merged

    def tfidf(self):
        if self.dl.pickle_exists('merged_tfidf'):
            print('Loading merged_tfidf pickle.')
            merged = self.dl.load_pickle('merged_tfidf')
        else:
            print('Creating merged_tfidf dataset.')
            merged = self.merged.copy()

            # TF-IDF and Truncated SVD initialization (to extract concept using Latent semantics analysis(LSA))
            tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')  # stop words
            # n_components=100 to extract concepts using LSA
            svd = TruncatedSVD(n_components=100, random_state=2019)

            # Creating pipeline to execute TF-IDF and SVD in one step
            pipe = Pipeline(steps=[('tfidf', tfidf), ('svd', svd)])

            merged["prod_desc_merge"] = merged["product_description"].map(
                str) + ' ' + merged["brand"].fillna('').map(str)

            # Perform fit and transform function of pipeline to convert text(in each feature) into vectors and reducing them
            merged["product_title"] = pipe.fit_transform(merged["product_title"])
            merged["search_term"] = pipe.fit_transform(merged["search_term"])
            merged["prod_desc_merge"] = pipe.fit_transform(merged["prod_desc_merge"])

            self.dl.save_pickle(merged, 'merged_tfidf')

        # Drop tring values that are not needed for the model
        merged.drop(['product_uid', "product_description", "brand"], axis=1, inplace=True)

        return merged

    def word2vec(self, sg=0, vector_size=300, min_count=2, window=5, workers=4):
        '''
        Create a Woc2Vec representation
        Output:
            A pandas dataframe
        '''
        id = f'_{sg}-{vector_size}-{min_count}-{window}'
        if self.dl.pickle_exists(f'merged_w2v{id}'):
            print(f'Loading merged_w2v{id} pickle.')
            merged = self.dl.load_pickle(f'merged_w2v{id}')
        else:
            print(f'Creating merged_w2v{id} dataset.')
            merged = self.merged.copy()
            w2v_model = self.word2vec_model(merged, id, sg, vector_size, min_count, window, workers)

            # Combile title, description and brand into a single string
            print('  Calculating sentence-level embeddings.')
            merged['combined'] = merged['product_title'] + \
                merged['product_description'] + merged['brand']
            merged['search_vector'] = merged['search_term'].apply(
                lambda x: self.get_embedding_w2v(x.split(), w2v_model))
            merged['title_vector'] = merged['product_title'].apply(
                lambda x: self.get_embedding_w2v(x.split(), w2v_model))
            merged['description_vector'] = merged['product_description'].apply(
                lambda x: self.get_embedding_w2v(x.split(), w2v_model))
            merged['brand_vector'] = merged['brand'].apply(
                lambda x: self.get_embedding_w2v(x.split(), w2v_model))
            merged['combined_vector'] = merged['combined'].apply(
                lambda x: self.get_embedding_w2v(x.split(), w2v_model))

            merged = self.add_cosine_sim(merged)

            # Save to pickle
            self.dl.save_pickle(merged, f'merged_w2v{id}')

        # Drop text unnecessary columns
        merged.drop(['id', 'product_uid', 'product_title', 'search_term',
                     'product_description', 'brand', 'combined', 'search_vector', 'title_vector', 'combined_vector', 'description_vector', 'brand_vector'], axis=1, inplace=True)

        return merged

    def add_cosine_sim(self, df):
        '''
        Create new columns with cosine similarity between columns
        Args & Output:
            A pandas DataFrame
        '''
        sim_term_title = []
        sim_term_desc = []
        sim_term_brand = []
        sim_term_combined = []

        # Iterate over all rows and compute cosine similarity between columns
        print('  Calculating cosize similarity between sentence embeddings.')
        for i, row in tqdm(df.iloc[:].iterrows()):

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

        df['sim_term_title'] = pd.Series(sim_term_title)
        df['sim_term_desc'] = pd.Series(sim_term_desc)
        df['sim_term_brand'] = pd.Series(sim_term_brand)
        df['sim_term_combined'] = pd.Series(sim_term_combined)

        return df

    def word2vec_model(self, df, id, sg, vector_size, min_count, window, workers):
        '''
        Create a Woc2Vec model
        '''
        # If word2vec models exists, load it
        if os.path.exists(f'{self.model_dir}word2vec{id}.model'):
            print(f'  Loading word2vec{id} model.')
            w2v_model = Word2Vec.load(f'{self.model_dir}word2vec{id}.model')
        else:
            print(f'  Building word2vec{id} model.')
            combined_training = pd.concat(
                [df.product_title, df.product_description, df.search_term, df.brand])

            train_data = []
            for i in combined_training:
                train_data.append(i.split())

            w2v_model = Word2Vec(train_data, vector_size=vector_size,
                                 min_count=min_count, window=window, sg=sg, workers=workers)
            w2v_model.save(f'{self.model_dir}word2vec{id}.model')
        return w2v_model

    # Function returning vector representation of a document
    def get_embedding_w2v(self, doc_tokens, w2v_model):
        '''
        Get the vector representation of a document based on word embeddings
        '''
        vector_size = w2v_model.vector_size
        embeddings = []
        if len(doc_tokens) < 1:
            return np.zeros(vector_size)
        else:
            for tok in doc_tokens:
                if tok in w2v_model.wv:
                    embeddings.append(w2v_model.wv.get_vector(tok, norm=True))
                else:
                    embeddings.append(np.random.rand(vector_size))
            # mean the vectors of individual words to get the vector of the document
            return np.mean(embeddings, axis=0)

    def doc2vec(self, dm=0, vector_size=300, min_count=2, window=5, workers=4):
        '''
        Create a Doc2Vec representation
        Output:
            A pandas dataframe
        '''
        id = f'_{dm}-{vector_size}-{min_count}-{window}'
        if self.dl.pickle_exists(f'merged_d2v{id}'):
            print(f'Loading merged_d2v{id} pickle.')
            merged = self.dl.load_pickle(f'merged_d2v{id}')
        else:
            print(f'Creating merged_d2v{id} dataset.')
            merged = self.merged.copy()
            # d2v_model = self.doc2vec_model(merged, id, dm, vector_size, min_count, window, workers)

            # If word2vec models exists, load it
            if os.path.exists(f'{self.model_dir}doc2vec{id}.model'):
                print(f'  Loading doc2vec{id} model.')
                d2v_model = Doc2Vec.load(f'{self.model_dir}doc2vec{id}.model')
            else:
                print(f'  Building doc2vec{id} model.')

                combined_training = pd.concat(
                    [merged.product_title, merged.product_description, merged.search_term, merged.brand])

                prefix = 'all'
                train_data = []
                for i, t in zip(combined_training.index, combined_training):
                    train_data.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))

                d2v_model = Doc2Vec(train_data, vector_size=vector_size,
                                    min_count=min_count, window=window, dm=dm, workers=workers)
                d2v_model.save(f'{self.model_dir}doc2vec{id}.model')

            # Combile title, description and brand into a single string
            print('  Calculating sentence-level embeddings.')

            # Create new column
            merged['combined'] = merged['product_title'] + \
                merged['product_description'] + merged['brand']

            # Get embeddings per column
            merged['search_vector'] = self.get_embedding_d2v(merged['search_term'], d2v_model)
            merged['title_vector'] = self.get_embedding_d2v(merged['product_title'], d2v_model)
            merged['description_vector'] = self.get_embedding_d2v(
                merged['product_description'], d2v_model)
            merged['brand_vector'] = self.get_embedding_d2v(merged['brand'], d2v_model)
            merged['combined_vector'] = self.get_embedding_d2v(merged['combined'], d2v_model)

            merged = self.add_cosine_sim(merged)

            # Save to pickle
            self.dl.save_pickle(merged, f'merged_d2v{id}')

        # Drop text unnecessary columns
        merged.drop(['id', 'product_uid', 'product_title', 'search_term',
                     'product_description', 'brand', 'combined', 'search_vector', 'title_vector', 'combined_vector', 'description_vector', 'brand_vector'], axis=1, inplace=True)

        return merged

    def get_embedding_d2v(self, docs, d2vmodel):
        vecs = []

        for i in docs.index:
            prefix = 'all_' + str(i)
            vecs.append(d2vmodel.dv[prefix])

        return pd.Series(vecs)
