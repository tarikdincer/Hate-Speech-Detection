import pandas as pd
from stop_words import get_stop_words
import numpy as np
import math
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from gensim.models import KeyedVectors
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import re
from TurkishStemmer import TurkishStemmer

def get_train_data(train_path):
    train_data = pd.read_csv(train_path, sep='\t')
    train_data.rename(columns = {'subtask_a' : 'label'}, inplace = True)
    train_data.loc[train_data["label"] == "NOT", "label"] = 0
    train_data.loc[train_data["label"] == "OFF", "label"] = 1
    train_data.rename(columns={'tweet': 'text', 'label': 'labels'}, inplace = True)
    train_data.drop('id', inplace=True, axis=1)

    return train_data

def preprocess_train_data(train_df):
    train_df["text"] = train_df["text"].str.lower()
    train_df["text"] = train_df["text"].str.replace(r"https?://[ˆ ]+ | www.[ˆ ]+", "")
    train_df["text"] = train_df["text"].str.replace(r"@[A-Za-z0-9]+", "")
    train_df["text"] = train_df["text"].str.replace(r"#[A-Za-z0-9]+", "")
    stop_words = get_stop_words('tr')

    for stop in stop_words:
        regex_1 = r'\b{0}\b'.format(stop)
        regex_2 = r'[^\w\s]'
        train_df["text"] = train_df["text"].str.replace(regex_1, "")
        train_df["text"] = train_df["text"].str.replace(regex_2, "")
    
    train_df["text"] = train_df["text"].str.replace(r" +", " ")

    train_df['tokens'] = ''
    train_df['tokens'] = train_df['tokens'].apply(list)
    stemmer = TurkishStemmer()

    for row in train_df.index.values:
        for word in train_df.at[row,"text"].split():
            train_df.at[row,"tokens"].append(stemmer.stem(word))

    return train_df

def gensim_to_keras_embedding(model, train_embeddings=False):
    keyed_vectors = model  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array    
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer


if __name__ == "__main__":
    train_df = get_train_data("./data/train/train.tsv")
    train_df = preprocess_train_data(train_df)
    # print(train_df)
    
    model = word2vec.Word2Vec(sentences=train_df["tokens"].tolist(),vector_size = 300, sg=1,workers=4)
    model.wv.save_word2vec_format('custom_word_embeddings.txt')

    # model = KeyedVectors.load_word2vec_format("custom_word_embeddings.txt", binary=False)
    # gensim_to_keras_embedding(model)

    # opt = optimizers.adam(lr=0.001)

    # model = Sequential()
    # model.add(gensim_to_keras_embedding(model))



    # model.add(SpatialDropout1D((0.25)))

    # model.add(Conv1D(filters=300, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2,stride=1))
    # model.add(BatchNormalization()) 
    # model.add(SpatialDropout1D((0.5)))

    # model.add(Conv1D(filters=300, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2,stride=1))
    # model.add(BatchNormalization()) 
    # model.add(SpatialDropout1D((0.5)))



    # model.add(Conv1D(300,4,padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    # model.add(advanced_activations.LeakyReLU(alpha=0.3))
    # model.add(MaxPooling1D(pool_size=2,stride=1))
    # model.add(BatchNormalization()) 
    # model.add(SpatialDropout1D((0.5)))

    # model.add(Conv1D(300,4,padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    # model.add(advanced_activations.LeakyReLU(alpha=0.3))
    # model.add(MaxPooling1D(pool_size=2,stride=1))
    # model.add(BatchNormalization()) 
    # model.add(SpatialDropout1D((0.5)))



    # model.add(Bidirectional(CuDNNLSTM(256,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),return_sequences=False)))


    # model.add(Dense(1,activation='sigmoid'))

    # model.compile(optimizer = opt,loss = 'binary_crossentropy',metrics =['accuracy'])
