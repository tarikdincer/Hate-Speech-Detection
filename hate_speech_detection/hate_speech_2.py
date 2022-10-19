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
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D, SpatialDropout1D, LeakyReLU, Bidirectional, LSTM
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import re
from TurkishStemmer import TurkishStemmer

def get_train_data(train_path):
    train_data = pd.read_csv(train_path, sep='\t')
    train_data.rename(columns = {'subtask_a' : 'label'}, inplace = True)
    # train_data.loc[train_data["label"] == "NOT", "label"] = 0
    # train_data.loc[train_data["label"] == "OFF", "label"] = 1
    train_data.rename(columns={'tweet': 'text', 'label': 'labels'}, inplace = True)
    train_data.drop('id', inplace=True, axis=1)

    return train_data

def preprocess_train_data(train_df):

    label_values=train_df.labels.unique()
    dic={}
    for i,label in enumerate(label_values):
        dic[label]=i
    labels=train_df.labels.apply(lambda x:dic[x])

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

    stemmer = TurkishStemmer()

    for row in train_df.index.values:
        temp_str = ""
        for word in train_df.at[row,"text"].split():
            temp_str += stemmer.stem(word) + " "
        train_df.at[row,"text"] = temp_str.strip()

    val_df=train_df.sample(frac=0.2,random_state=200)
    train_df=train_df.drop(val_df.index)

    texts = train_df.text

    NUM_WORDS=20000
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid=tokenizer.texts_to_sequences(val_df.text)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X_train = pad_sequences(sequences_train)
    X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])
    y_train = to_categorical(np.asarray(labels[train_df.index]))
    y_val = to_categorical(np.asarray(labels[val_df.index]))
    print('Shape of X train and X validation tensor:', X_train.shape,X_val.shape)
    print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)

    return X_train, X_val, y_train, y_val,word_index

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
    X_train, X_val, y_train, y_val,word_index = preprocess_train_data(train_df)
    
    # model = word2vec.Word2Vec(sentences=train_df["tokens"].tolist(), 
    #              sg=1,
    #              workers=4)
    # model.wv.save_word2vec_format('custom_word_embeddings.txt')

    word_vectors = KeyedVectors.load_word2vec_format("custom_word_embeddings.txt", binary=False)

    EMBEDDING_DIM=64
    vocabulary_size=min(len(word_index)+1,20000)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i>=20000:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

    del(word_vectors)

    embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
    

    opt = Adam(lr=0.001)

    model = Sequential()
    model.add(embedding_layer)


    model.add(SpatialDropout1D((0.25)))

    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1))
    model.add(BatchNormalization()) 
    model.add(SpatialDropout1D((0.5)))


    model.add(Conv1D(64,4,padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2,strides=1))
    model.add(BatchNormalization()) 
    model.add(SpatialDropout1D((0.5)))



    model.add(Bidirectional(LSTM(32,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),return_sequences=False)))


    model.add(Dense(2,activation='softmax'))

    model.compile(optimizer = opt,loss = 'categorical_crossentropy',metrics =['accuracy'])
    checkpoint = ModelCheckpoint('model{epoch:08d}.h5', period=5) 
    model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_val, y_val),
         callbacks=checkpoint)
