# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:06:07 2020

@author: Jonathan
"""


from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import datetime 
from tensorflow.keras.preprocessing.text import Tokenizer
import math
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


#read in and prepare text data
dfText = pd.read_csv('data/dailyRedpolitics.csv')
dfText= dfText.drop([0]).reset_index(drop=True)
dfText.afterEpoch = dfText.afterEpoch.astype(int)
dfText.beforeEpoch = dfText.beforeEpoch.astype(int)
#dfText = dfText.replace(np.nan,'')
dfText['sentence'] = dfText.art0 + '. ' + dfText.art1+ '. ' + dfText.art2+ '. ' + dfText.art3+ '. ' + dfText.art4+ '. ' + dfText.art5 + '. ' + dfText.art6+ '. ' + dfText.art7+ '. ' + dfText.art8+ '. ' + dfText.art9+ '. ' + dfText.art10
dfText['sentence'] = dfText['sentence'].astype('string')

#read in and prepare VIX data
dfVix = pd.read_csv('data/VIX_ALL_yahoo.csv')
dfVix = dfVix.iloc[4938:7718].reset_index(drop=True)

#read in and prepare SPY data
dfSpy = pd.read_csv('data/SPY_ALL_yahoo.csv')
dfSpy = dfSpy.iloc[4159:6939].reset_index(drop=True)
dfSpy['vixOpen'] = dfVix['Open']
dfSpy['vixClose'] = dfVix['Close']
dfSpy['date1'] = pd.to_datetime(dfSpy['Date'])
dfSpy['startDate'] = (dfSpy['date1'] - datetime.datetime(1970,1,1)).dt.total_seconds().astype(int)
min30 = 1800
dfSpy['startDate'] = dfSpy['startDate'] + (27*min30)
dfSpy['endDate'] = dfSpy['startDate'] + (13*min30)
dfSpy['large'] = 0
dfSpy['small'] = 0
dfSpy['magnitude'] = abs((dfSpy.Close-dfSpy.Open)/dfSpy.Open)/((dfSpy.vixOpen/100)/math.sqrt(500))
dfSpy.loc[(dfSpy.magnitude >= 0.5), 'large'] = 1
dfSpy.loc[(dfSpy.magnitude < 0.5), 'small'] = 1
dfText['large'] = dfSpy['large']
dfText['small'] = dfSpy['small']

#shuffle and Train/Test split data
dfText = dfText.sample(frac=1)
X_train, X_test, y_train, y_test = train_test_split(dfText['sentence'], dfText[['large', 'small']], test_size=0.20, random_state=42)

#fit tokenizer on text
t = Tokenizer()
t.fit_on_texts(X_train)

#look at tokenizer info
z=t.word_counts
z=t.word_index
z=t.word_docs

#tokenize, truncate, and pad text
encodedText = t.texts_to_sequences(X_train)
max_length = 100
pad_trunc_Text = pad_sequences(encodedText, maxlen=max_length, padding='post', truncating='post')
vocab_size = len(t.word_index)+1

#load GloVe embeddings
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('word vectors = {}'.format(len(embeddings_index)))

#create a weight matrix with training data for embedding layer
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

#build model using GloVe embeddings and freeze embedding layer intially 
inputs = tf.keras.Input(shape=(max_length,))
x = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=200, trainable=False, mask_zero=True)(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dropout(.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
out = tf.keras.layers.Dense(2, activation="softmax")(x)

#choose optimizer
opt = Adam(2e-5)
#opt = SGD(lr=1e-4, momentum=0.0,nesterov=True)

#compile and train model with embedding layer frozen
model = tf.keras.Model(inputs=inputs, outputs=out)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
history = model.fit(pad_trunc_Text,y_train,epochs=5,batch_size=32,validation_split=0.2,shuffle=True)
model.summary()

#unfreeze embedding layer
for layer in model.layers[:]:
    print(layer, layer.trainable)

for layer in model.layers[1:2]:
    layer.trainable = True

for layer in model.layers[:]:
    print(layer, layer.trainable)

#decrease learning rate, compile, and train model again for a few more epochs
opt = Adam(2e-7)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
history = model.fit(pad_trunc_Text,y_train,epochs=5,batch_size=32,validation_split=0.2,shuffle=True)
model.summary()

#evaluate model
def evalModel():
    encodedTextTest = t.texts_to_sequences(X_test)
    pad_trunc_TextTest = pad_sequences(encodedTextTest, maxlen=max_length, padding='post', truncating='post')
    u = model.predict(pad_trunc_TextTest)
    v = np.argmax(u, axis=1)
    w = np.asarray(y_test)
    w = np.argmax(w, axis=1)
    print(confusion_matrix(w,v))
    print("Accuracy:", accuracy_score(w,v))
    print("F1:", f1_score(w,v))
    print("Precision:", precision_score(w,v))
    print("Recall:", recall_score(w,v))

def plotLoss():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('tain loss vs val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def plotAcc():
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('train acc vs val acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    

plotLoss()
plotAcc()
evalModel()

