# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:44:25 2020

@author: Jonathan
"""

from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#read in and prepare reddit data 
dfText = pd.read_csv('data/redpolitics30min.csv', index_col=0)
dfText= dfText[130:].reset_index(drop=True)
dfText.afterEpoch = dfText.afterEpoch.astype(int)
dfText.beforeEpoch = dfText.beforeEpoch.astype(int)
dfText = dfText.replace(np.nan,'')
dfText['sentence'] = dfText.art0 + '. ' + dfText.art1+ '. ' + dfText.art2+ '. ' + dfText.art3+ '. ' + dfText.art4+ '. ' + dfText.art5 + '. ' + dfText.art6+ '. ' + dfText.art7+ '. ' + dfText.art8+ '. ' + dfText.art9
dfText['sentence'] = dfText['sentence'].astype('string')

#read in and prepare spy data
dfSpy = pd.read_csv('data/spy_30Min2yrsNum2.csv', index_col=0)
dfSpy = dfSpy[:6103].reset_index(drop=True)

#create up/down target column
dfSpy['up'] = 0
dfSpy['down'] = 0
dfSpy['direction'] = np.log(dfSpy.close/dfSpy.opn)
dfSpy.loc[(dfSpy.direction >= 0), 'up'] = 1
dfSpy.loc[(dfSpy.direction < 0), 'down'] = 1
#one column label
dfSpy['directionTarget1Col'] = 0
dfSpy.loc[(dfSpy.direction >= 0), 'directionTarget1Col'] = 1

#replace missing volumes with median
volMed = dfSpy['volume'].median()
dfSpy['volume'] = dfSpy['volume'].replace(0,volMed)

#shift columns to create features from prior/next day
dfSpy['close_open'] = np.log(dfSpy.close/dfSpy.opn).shift(1)
dfSpy['high_open'] = np.log(dfSpy.high/dfSpy.opn).shift(1)
dfSpy['low_open'] = np.log(dfSpy.opn/dfSpy.low).shift(1)
for i in range(len(dfSpy['volume'])-1):
    dfSpy.loc[i+1,'volPercent'] = np.log(dfSpy.loc[i+1,'volume']/dfSpy.loc[i,'volume'])

#add combined sentence feature and drop first row
dfSpy['sentence']=dfText['sentence']
dfSpy= dfSpy.drop([0]).reset_index(drop=True)

#create final preNormalized feature/target df
FinDf=dfSpy[['close_open', 'high_open', 'low_open', 'volPercent', 
             'sentence', 'directionTarget1Col']]

#separate df into train, val, test
dftrainDataWith_y = FinDf[:3900]
dfvalDataWith_y = FinDf[3900:4880]
dftestDataWith_y = FinDf[4880:]

#fit and standardize train data for numerical input
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtrain = scaler.fit_transform(dftrainDataWith_y[['close_open', 'high_open', 'low_open', 'volPercent', ]])

#normalize val and test data for numerical input
Xval = scaler.transform(dfvalDataWith_y[['close_open', 'high_open', 'low_open', 'volPercent', ]])
Xtest = scaler.transform(dftestDataWith_y[['close_open', 'high_open', 'low_open', 'volPercent', ]])

#set Target column and target values
target = 'directionTarget1Col'
ytrain = dftrainDataWith_y[target]
yval = dfvalDataWith_y[target]
ytest = dftestDataWith_y[target]

#tokenize/encode/pad train data for text input
t = Tokenizer()
t.fit_on_texts(dftrainDataWith_y['sentence'])
max_length = 100
vocab_size = len(t.word_index)+1
encodedTrain = t.texts_to_sequences(dftrainDataWith_y['sentence'])
padded_encodedTrain = pad_sequences(encodedTrain, maxlen=max_length, padding='post', truncating='post')

#tokenize/encode/pad val data for text input
encodedVal = t.texts_to_sequences(dfvalDataWith_y['sentence'])
padded_encodedVal = pad_sequences(encodedVal, maxlen=max_length, padding='post', truncating='post')

#tokenize/encode/pad test data for text input
encodedTest = t.texts_to_sequences(dftestDataWith_y['sentence'])
padded_encodedTest = pad_sequences(encodedTest, maxlen=max_length, padding='post', truncating='post')

#shuffle everything
from sklearn.utils import shuffle
Xtrain, padded_encodedTrain, ytrain = shuffle(Xtrain, padded_encodedTrain, ytrain, random_state=37)
Xval, padded_encodedVal, yval = shuffle(Xval, padded_encodedVal, yval, random_state=37)
Xtest, padded_encodedTest, ytest = shuffle(Xtest, padded_encodedTest, ytest, random_state=37)

#weighting target classes for loss function
'''
#create weights for target column
sumUp=sum(dftrainDataWith_y[target])
length = len(dftrainDataWith_y[target])
weight=sumUp/(length-sumUp)
class_weight = {0: weight, 1: 1.0}
'''

#build model architecture
n_features = Xtrain.shape[1]

inputsNum = tf.keras.Input(shape=(n_features,))
inputsText = tf.keras.Input(shape=(max_length,))

x1 = tf.keras.layers.Dense(5, activation='relu')(inputsNum)
x1 = tf.keras.layers.Dropout(.4)(x1)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.Dense(2, activation='relu')(x1)

x2 = tf.keras.layers.Embedding(vocab_size, 50)(inputsText)
x2 = tf.keras.layers.Dropout(.4)(x2)
x2 = tf.keras.layers.BatchNormalization()(x2)
x2 = tf.keras.layers.Flatten()(x2)
x2 = tf.keras.layers.Dense(2, activation='relu')(x2)
x2 = tf.keras.layers.Dropout(.4)(x2)

xAll = tf.keras.layers.concatenate([x1,x2])
xAll = tf.keras.layers.Dense(2)(xAll)

out = tf.keras.layers.Dense(1, activation="sigmoid")(xAll)

#hyperparameters
opt = Adam(2e-5)
#opt = SGD(lr=0.001)
#loss = tf.keras.losses.CategoricalCrossentropy()
loss = 'binary_crossentropy'

#compile and fit model
model = tf.keras.Model(inputs=[inputsNum, inputsText], outputs=out)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
history = model.fit([Xtrain, padded_encodedTrain], ytrain, epochs=55, batch_size = 32, shuffle=(True),
                    validation_data = ([Xval, padded_encodedVal], yval))
model.summary()

#evaluate model
def evalModel():
    v = np.round(model.predict([Xtest, padded_encodedTest]))
    w = ytest
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
    
evalModel()
plotLoss()
plotAcc()
