
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
from sklearn.metrics import precision_recall_fscore_support as score


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
    train_x.append(data.iloc[i,2])
    x.append(data.iloc[i,2])
    
    if ratting==1:
      train_y.append(0)
    elif ratting==2:
      train_y.append(1)
    elif ratting==3:
      train_y.append(2)
    elif ratting==4:
      train_y.append(3)
    elif ratting==5:
      train_y.append(4)
      
       
  
  row,col = data1.shape
  
  i=0
  for i in range(int(size/10)):
    ratting = data1.iloc[i,0]
    test_x.append(data1.iloc[i,2])
    x.append(data1.iloc[i,2])
    
    if ratting==1:
      test_y.append(0)
    elif ratting==2:
      test_y.append(1)
    elif ratting==3:
      test_y.append(2)
    elif ratting==4:
      test_y.append(3)
    elif ratting==5:
      test_y.append(4)
      
  return len(train_y), len(test_y)





def text_process(mess):

    nopunc = ''
    for ch in range(len(mess)):
        if ((mess[ch] == ' ') or (mess[ch] not in string.punctuation)):
            nopunc = nopunc+ mess[ch]
   
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
def train_model(classifier, feature_vector_train, label, feature_vector_test, tt_y,is_neural_net=False):

    classifier.fit(feature_vector_train, label)
 
    predictions = classifier.predict(feature_vector_test)
    
    
    if(is_neural_net):
      y_pred = np.argmax(predictions,axis=1)
      y_test = np.argmax(tt_y,axis=1)
      
      return  accuracy(y_pred,y_test),score(y_pred,y_test,average = 'micro')
    
    else:
      print(predictions,tt_y)
      return accuracy(predictions,tt_y),score(predictions,tt_y, average = 'micro')

def accuracy(y_pred,y_test):
  count =0 
  for i in range(len(y_pred)):
    
    if(y_pred[i]==y_test[i]):
      count = count+1
      
  return (count/len(y_test))

 
def create_cnn():

    input_layer = layers.Input((70, ))

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(5, activation="softmax")(output_layer1)

    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
    
    return model
 

def create_rnn_lstm():
 
    input_layer = layers.Input((70, ))

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)


    lstm_layer = layers.LSTM(100)(embedding_layer)


    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(5, activation="softmax")(output_layer1)

    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
    
    return model
    
    
def create_rnn_gru():
  
    input_layer = layers.Input((70, ))


    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    lstm_layer = layers.GRU(100)(embedding_layer)

    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(5, activation="softmax")(output_layer1)
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
    
    return model
  
    


embeddings_index = {}
word_index = []
for i, line in enumerate(open('wiki-news-300d-1M.vec',encoding='UTF8')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    
    

heading = ['Train','Test','LR_Acc','LR_Precision','LR_Recall','LR_F','SVM_Acc','SVM_Precision','SVM_Recall','SVM_F','NB_Acc','NB_Precision','NB_Recall','NB_F','CNN_Acc','CNN_Precision','CNN_Recall','CNN_F','LSTM_Acc','LSTM_Precision','LSTM_Recall','LSTM_F','GRU_Acc','GRU_Precision','GRU_Recall','GRU_F']
file1 = open('results_5_class_all.csv','w')
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
  
  d_tr_y = np_utils.to_categorical(train_y)
  d_tt_y = np_utils.to_categorical(test_y)
  
  accuracy1, score1 = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,test_y)
  precision1, recall1, fscore1, support1 = score1[0],score1[1],score1[2],score1[3]
 
  accuracy2, score2 = train_model(svm.LinearSVC(C=1), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram,test_y)
  precision2, recall2, fscore2, support2 = score2[0],score2[1],score2[2],score2[3]
  
  accuracy3, score3 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram, test_y)
  precision3, recall3, fscore3, support3 = score3[0],score3[1],score3[2],score3[3]
  
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
  
  accuracy4, score4 = train_model(classifier, train_seq_x, d_tr_y, test_seq_x, d_tt_y, is_neural_net=True)
  precision4, recall4, fscore4, support4 = score4[0],score4[1],score4[2],score4[3]
 

  classifier = create_rnn_lstm()
  accuracy5,score5 = train_model(classifier, train_seq_x, d_tr_y, test_seq_x, d_tt_y,is_neural_net=True)
  precision5, recall5, fscore5, support5 = score5[0],score5[1],score5[2],score5[3]
  
  
  classifier = create_rnn_gru()
  accuracy6, score6 = train_model(classifier, train_seq_x, d_tr_y, test_seq_x, d_tt_y, is_neural_net=True)
  precision6, recall6, fscore6, support6 = score6[0],score6[1],score6[2],score6[3]
  
  
  
  row = [size, size/10, accuracy1, precision1, recall1, fscore1, accuracy2, precision2, recall2,fscore2, accuracy3, precision3, recall3, fscore3, accuracy4, precision4, recall4, fscore4,accuracy5, precision5, recall5, fscore5,accuracy6, precision6, recall6, fscore6 ]
  
  file1 = open('results_5_class_all.csv','a')
  wtr = csv.writer(file1)
  wtr.writerow(row)
  file1.close()
  x = []
  train_x = []
  train_y = []
  test_x = []
  test_y = []
  
  
  
  
  
  
  
