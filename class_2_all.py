
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
import string
import csv
from keras.utils import np_utils
    
data = pd.read_csv("train.csv",encoding='UTF8')
data1 = pd.read_csv("test.csv",encoding='UTF8')
x = []
train_x = []
train_y = []
test_x = []
test_y = []



def read_data(size):
  row,col = data.shape
  for i in range(size):
    ratting = data.iloc[i,0]
    
    x.append(data.iloc[i,2])
    if ratting != 3:
      train_x.append(data.iloc[i,2])
      if ratting<3:
        train_y.append(0)
      else:
        train_y.append(1)
       
  
  row,col = data1.shape
  
  i=0
  for i in range(int(size/10)):
    ratting = data1.iloc[i,0]
    
    x.append(data1.iloc[i,2])
    if ratting != 3:
      test_x.append(data1.iloc[i,2])
      if ratting<3:
        test_y.append(0)
      else:
        test_y.append(1)
      
  return len(train_y), len(test_y)


def text_process(mess):

    nopunc = ''
    for ch in range(len(mess)):
        if ((mess[ch] == ' ') or (mess[ch] not in string.punctuation)):
            nopunc = nopunc+ mess[ch]
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
def train_model(classifier, feature_vector_train, label, feature_vector_test,is_neural_net=False):
 
    classifier.fit(feature_vector_train, label)
    
   
    predictions = classifier.predict(feature_vector_test)
    
    return metrics.accuracy_score(np.round(predictions), test_y),metrics.precision_score(np.round(predictions), test_y),metrics.recall_score(np.round(predictions), test_y)


def create_rnn_lstm():
   
    input_layer = layers.Input((70, ))

  
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)


    lstm_layer = layers.LSTM(100)(embedding_layer)


    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

  
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model
 
def create_cnn():
   
    input_layer = layers.Input((70, ))

   
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

 
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

 
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
  
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model
    
def f_score(p,r):
  f = (2*p*r)/(p+r)
  return f
  
  
def create_rnn_gru():
 
    input_layer = layers.Input((70, ))

   
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)


    lstm_layer = layers.GRU(100)(embedding_layer)

   
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

   
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model
  
    


embeddings_index = {}
word_index = []


for i, line in enumerate(open('wiki-news-300d-1M.vec',encoding='UTF8')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
  

heading = ['Train','Test','LR_Acc','LR_Precision','LR_Recall','LR_F','SVM_Acc','SVM_Precision','SVM_Recall','SVM_F','NB_Acc','NB_Precision','NB_Recall','NB_F','CNN_Acc','CNN_Precision','CNN_Recall','CNN_F','LSTM_Acc','LSTM_Precision','LSTM_Recall','LSTM_F','GRU_Acc','GRU_Precision','GRU_Recall','GRU_F']
file1 = open('results_2_class_all.csv','w')
wtr = csv.writer(file1)
wtr.writerow(heading)
file1.close()
  
for j in range(50):

  size = 50000+j*50000
  actual_train_size, actual_test_size = read_data(size)
  
  tfidf_vect_ngram = TfidfVectorizer(analyzer=text_process, ngram_range=(2,3), max_features=5000)
  tfidf_vect_ngram.fit(x)
  xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
  xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)


 
  accuracy1, precision1, recall1 = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,False)
  

  accuracy2, precision2, recall2 = train_model(svm.LinearSVC(C=1), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,False)
  
 
  accuracy3, precision3, recall3 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram, False)
  
  

  token = text.Tokenizer()
  token.fit_on_texts(x)
  word_index = token.word_index
  
  train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
  test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=70)
  
  embedding_matrix = np.zeros((len(word_index) + 1, 300))
  for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
  
  
  

  classifier = create_cnn()
  accuracy4, precision4, recall4 = train_model(classifier, train_seq_x, train_y, test_seq_x, is_neural_net=True)

  classifier = create_rnn_lstm()
  accuracy5, precision5, recall5  = train_model(classifier, train_seq_x, train_y, test_seq_x, is_neural_net=True)
 
  
  classifier = create_rnn_gru()
  accuracy6, precision6, recall6 = train_model(classifier, train_seq_x, train_y, test_seq_x, is_neural_net=True)
 
  
  
  row = [size, size/10, actual_train_size, actual_test_size, accuracy1, precision1, recall1, f_score(precision1,recall1), accuracy2, precision2, recall2,f_score(precision2,recall2), accuracy3, precision3, recall3, f_score(precision3,recall3), accuracy4, precision4, recall4, f_score(precision4,recall4), accuracy5, precision5, recall5, f_score(precision5,recall5),accuracy6, precision6, recall6, f_score(precision6,recall6)]

  file1 = open('results_2_class_all.csv','a')
  wtr = csv.writer(file1)
  wtr.writerow(row)
  file1.close()
  x = []
  train_x = []
  train_y = []
  test_x = []
  test_y = []
  
  
  
  
  
  
  
