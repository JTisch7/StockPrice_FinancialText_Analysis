# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:10:20 2021

@author: Jonathan
"""

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import math
import tqdm
import tensorflow as tf

#import stock price dfs
dfamzn = pd.read_csv('data/30MinAMZN2yrs.csv', index_col=(0))
dfaapl = pd.read_csv('data/30MinAAPL2yrs.csv', index_col=(0))
dfgoog = pd.read_csv('data/30MinGOOG2yrs.csv', index_col=(0))

#change dates to workable datetimes
for j in range(len(dfamzn)):
    dfamzn.loc[j,'date'] = parse(dfamzn['date'][j])

for j in range(len(dfaapl)):
    dfaapl.loc[j,'date'] = parse(dfaapl['date'][j])
    
for j in range(len(dfgoog)):
    dfgoog.loc[j,'date'] = parse(dfgoog['date'][j])

#import and prepare news df 
def prepNews(ticker):
    df = pd.read_csv('data/news'+str(ticker)+'2yrs.csv', index_col=(0))
    
    for j in range(len(df)):
        df['date'][j] = parse(df['date'][j])
    
    df = df.drop_duplicates(subset='title', keep='last')
    df = df[df.source != 'SEC']
    df['description'] = df['description'].replace(np.nan, '')
    return df.reset_index(drop=True)

dfamznNews = prepNews('AMZN')
dfaaplNews = prepNews('AAPL')
dfgoogNews = prepNews('GOOG')

#Use pretrained bert model to get sentiments for 'title' and 'description' of articles
from transformers import BertTokenizer, TFBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert', from_pt=True)
model = TFBertForSequenceClassification.from_pretrained("ProsusAI/finbert", from_pt=True)
#or 
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = tf.keras.models.load_model("my_FinBERT_96289")

def sentScore(dfNews):
    df = dfNews.copy()
    
    df['pos'] = 0
    df['neg'] = 0
    df['neut'] = 0
    df['pred'] = 0
    
    MAX_LEN = 160
    class_names = ['positive', 'negative', 'neutral']
    i = 0
    
    start = time.time()
    for sentence in tqdm(df['title']):
        encoded_new = tokenizer.encode_plus(
                                            sentence,                      
                                            add_special_tokens = True,       
                                            max_length = MAX_LEN,      
                                            truncation = True,
                                            padding = 'max_length',
                                            return_attention_mask = True,    
                                            return_tensors = 'tf',            
                                            )
         
        input_idst = (encoded_new['input_ids'])
        attention_maskst = (encoded_new['attention_mask'])
        
        new_test_output = model(input_idst, token_type_ids=None, 
                              attention_mask=attention_maskst)
        
        predicted = new_test_output[0].numpy()
        flat_predictions = np.concatenate(predicted, axis=0)
        new_predictions = np.argmax(flat_predictions).flatten()
        
        df.loc[i,'pos'] = predicted[0][0]
        df.loc[i,'neg'] = predicted[0][1]
        df.loc[i,'neut'] = predicted[0][2] 
        df.loc[i,'pred'] = class_names[new_predictions[0]] 
           
        i += 1
     
    df['posDes'] = 0
    df['negDes'] = 0
    df['neutDes'] = 0
    df['predDes'] = 0
    i = 0
    
    for sentence in df['description']:
        encoded_new = tokenizer.encode_plus(
                                sentence,                     
                                add_special_tokens = True,      
                                max_length = MAX_LEN,            
                                truncation = True,
                                pad_to_max_length = True,
                                return_attention_mask = True,     
                                return_tensors = 'tf',            
                           )
 
        input_idst = (encoded_new['input_ids'])
        attention_maskst = (encoded_new['attention_mask'])
        
        new_test_output = model(input_idst, token_type_ids=None, 
                              attention_mask=attention_maskst)
        
        predicted = new_test_output[0].numpy()
        flat_predictions = np.concatenate(predicted, axis=0)
        new_predictions = np.argmax(flat_predictions).flatten()
        
        df.loc[i,'posDes'] = predicted[0][0]
        df.loc[i,'negDes'] = predicted[0][1]
        df.loc[i,'neutDes'] = predicted[0][2] 
        df.loc[i,'predDes'] = class_names[new_predictions[0]] 
         
        i += 1
             
    finish = time.time()
    totalTime = (finish - start)/60
    print(totalTime)
    return df
    
dfamznNewsWsent = sentScore(dfamznNews)
dfaaplNewsWsent = sentScore(dfaaplNews)
dfgoogNewsWsent = sentScore(dfgoogNews)

'''
dfamznNewsWsent = pd.read_csv('data/amznNewsWsent.csv')
dfaaplNewsWsent = pd.read_csv('data/aaplNewsWsent.csv')
dfgoogNewsWsent = pd.read_csv('data/googNewsWsent.csv')
'''

#create empty df and sort articles for each time step then get average
def dividingFunc(dfNewsWsent, dfStock):
    NewsSortedTitle = np.zeros(shape=(len(dfamzn),500,3))
    NewsSortedDescrip = np.zeros(shape=(len(dfamzn),500,3))
    
    for i in range(len(dfStock)-1):
        k = 0
        for j in range(len(dfNewsWsent)):
            if (parse(dfNewsWsent['date'][j]) < dfStock['date'][i+1]) and (parse(dfNewsWsent['date'][j]) > dfStock['date'][i]):
                NewsSortedTitle[i,k]=np.array([dfNewsWsent['pos'][j],dfNewsWsent['neut'][j],dfNewsWsent['neg'][j]])
                NewsSortedDescrip[i,k]=np.array([dfNewsWsent['posDes'][j],dfNewsWsent['neutDes'][j],dfNewsWsent['negDes'][j]])
                k+=1
        print(i)
    
    #get averages of sentiments to use as features
    tCnt = np.count_nonzero(NewsSortedTitle, axis=1)
    tSum = np.sum(NewsSortedTitle, axis=1)
    tAve = tSum/tCnt
    
    dCnt = np.count_nonzero(NewsSortedDescrip, axis=1)
    dSum = np.sum(NewsSortedDescrip, axis=1)
    dAve = dSum/dCnt
    
    #tAve = pd.DataFrame(tAve)
    #dAve = pd.DataFrame(dAve)
    
    return tAve, dAve, tCnt, dCnt, NewsSortedTitle, NewsSortedDescrip

amznSentInfo = dividingFunc(dfamznNewsWsent, dfamzn)
aaplSentInfo = dividingFunc(dfaaplNewsWsent, dfaapl)
googSentInfo = dividingFunc(dfgoogNewsWsent, dfgoog)

'''
amznSentInfo = pd.read_csv('data/amznInfo.csv', index_col=0)
aaplSentInfo = pd.read_csv('data/aaplInfo.csv', index_col=0)
googSentInfo = pd.read_csv('data/googInfo.csv', index_col=0)

def convertBack(df):
    tAve = df[['tAvePos','tAveNeg','tAveNeut']].to_numpy()
    dAve = df[['dAvePos','dAveNeg','dAveNeut']].to_numpy()
    Cnt = np.reshape(np.array(df['Cnt']),(-1,1))
    return [tAve,dAve,Cnt]

amznSentInfo = convertBack(amznSentInfo)
aaplSentInfo = convertBack(aaplSentInfo)
googSentInfo = convertBack(googSentInfo)
'''

#import stock vix's
vxamzn = pd.read_csv('data/vxamzndailyprices.csv', header=1)
vxamzn = vxamzn[2180:].reset_index(drop=True)

vxaapl = pd.read_csv('data/vxaapldailyprices.csv', header=1)
vxaapl = vxaapl[2180:].reset_index(drop=True)

vxgoog = pd.read_csv('data/vxgoogdailyprices.csv', header=1)
vxgoog = vxgoog[2180:].reset_index(drop=True)

#add vix and sentiment features to 30min stock df
def addingDailyFeatures(vxStock, dfStock, tAve, dAve, tCnt):
    dfStock = dfStock.copy()
    dfStock.loc[:,'vixOpen'] = 0
    dfStock.loc[:,'vixHigh'] = 0
    dfStock.loc[:,'vixLow'] = 0
    dfStock.loc[:,'vixClose'] = 0
    for i in range(len(dfStock)):
        for j in range(len(vxStock)):
            if dfStock.loc[i,'date'].date()==parse(vxStock.loc[j,'Date']).date():
                dfStock.loc[i,'vixOpen'] = vxStock.loc[j,'Open']
                dfStock.loc[i,'vixHigh'] = vxStock.loc[j,'High']
                dfStock.loc[i,'vixLow'] = vxStock.loc[j,'Low']
                dfStock.loc[i,'vixClose'] = vxStock.loc[j,'Close']
                print(i)
                break
    dfStock['tPos'] = tAve[:,0]
    dfStock['tNeg'] = tAve[:,1]
    dfStock['tNeut'] = tAve[:,2]
    dfStock['dPos'] = dAve[:,0]
    dfStock['dNeg'] = dAve[:,1]
    dfStock['dNeut'] = dAve[:,2]
    dfStock['artCnt'] = tCnt[:,0]
    return dfStock

dfamzn = addingDailyFeatures(vxamzn, dfamzn, amznSentInfo[0], amznSentInfo[1], amznSentInfo[2])
dfaapl= addingDailyFeatures(vxaapl, dfaapl, aaplSentInfo[0], aaplSentInfo[1], aaplSentInfo[2])
dfgoog = addingDailyFeatures(vxgoog, dfgoog, googSentInfo[0], googSentInfo[1], googSentInfo[2])

#add weekday features to df
def addWeekday(dfStock):
    df = dfStock.copy()
    df['weekday'] = 0
    for i in range(len(df['date'])):
        df.loc[i,'weekday'] = df['date'][i].weekday()
    return df

dfamzn = addWeekday(dfamzn)
dfaapl = addWeekday(dfaapl)
dfgoog = addWeekday(dfgoog)

#read in spy and vix
dfspy = pd.read_csv('data/30MinSPY2yrs.csv', index_col=(0))
vxspy = pd.read_csv('data/VIX_NEW_YAHOO.csv')
vxspy = vxspy[743:].reset_index(drop=True)

#add vix to 30 min spy
def addingDailyVixToSpy(vxStock, dfStock):
    dfStock = dfStock.copy()
    dfStock.loc[:,'vixOpen'] = 0
    dfStock.loc[:,'vixHigh'] = 0
    dfStock.loc[:,'vixLow'] = 0
    dfStock.loc[:,'vixClose'] = 0
    for i in range(len(dfStock)):
        for j in range(len(vxStock)):
            if parse(dfStock.loc[i,'date']).date()==parse(vxStock.loc[j,'Date']).date():
                dfStock.loc[i,'vixOpen'] = vxStock.loc[j,'Open']
                dfStock.loc[i,'vixHigh'] = vxStock.loc[j,'High']
                dfStock.loc[i,'vixLow'] = vxStock.loc[j,'Low']
                dfStock.loc[i,'vixClose'] = vxStock.loc[j,'Close']
                print(i)
                break
    return dfStock

dfspyWvix = addingDailyVixToSpy(vxspy, dfspy)

#Create return features and target columns
def creatingReturnFeat(dfStock):
    df = dfStock.copy()
    #create vix-based return features
    df.loc[df.volume==0,'volume']=df['volume'].mean()
    for i in range(len(df.vixOpen)):
        df.loc[i,'vixRand'] = (np.random.random()*(df.loc[i,'vixHigh']-df.loc[i,'vixLow']))+df.loc[i,'vixLow']
    df['vixAve'] = (df.vixHigh+df.vixLow)/2
    vixTimestep = 14000
    df['close_open'] = np.log(df.close/df.opn)/((df.vixRand/100)/math.sqrt(vixTimestep))
    df['high_open'] = np.log(df.high/df.opn)/((df.vixRand/100)/math.sqrt(vixTimestep))
    df['low_open'] = np.log(df.opn/df.low)/((df.vixRand/100)/math.sqrt(vixTimestep))
    #create vix-based magnitude target
    df['large'] = 0
    df['small'] = 0
    df['magnitude'] = abs(np.log(df.close/df.opn))/((df.vixRand/100)/math.sqrt(vixTimestep))
    df.loc[(df.magnitude >= 0.675), 'large'] = 1
    df.loc[(df.magnitude < 0.675), 'small'] = 1
    for i in range(len(df['volume'])-1):
        df.loc[i+1,'volPercent'] = np.log(df.loc[i+1,'volume']/df.loc[i,'volume'])
    df['vixClose'] = df.vixClose/100
    #create direction target
    df['up'] = 0
    df['down'] = 0
    df.loc[(df.close_open >= 0), 'up'] = 1
    df.loc[(df.close_open < 0), 'down'] = 1 
    return df

dfamzn = creatingReturnFeat(dfamzn)
dfaapl = creatingReturnFeat(dfaapl)
dfgoog = creatingReturnFeat(dfgoog)
dfspy = creatingReturnFeat(dfspyWvix)

#add spy to dfs and drop rows
def addSpy(dfStock, dfSpy=dfspy):
    dfStock = dfStock.copy()
    dfSpy = dfSpy.copy()
    dfStock['spy_close_open'] = dfSpy['close_open']
    dfStock['spy_high_open'] = dfSpy['high_open']
    dfStock['spy_low_open'] = dfSpy['low_open']
    dfStock['spy_vix'] = dfSpy['vixClose']
    return dfStock

dfamzn = addSpy(dfamzn)
dfaapl = addSpy(dfaapl)
dfgoog = addSpy(dfgoog)

#shift target column and drop last row 
def shiftTarget(dfStock):
    df = dfStock.copy()
    df['large'] = df['large'].shift(-1)
    df['small'] = df['small'].shift(-1)
    df['up'] = df['up'].shift(-1)
    df['down'] = df['down'].shift(-1)        
    return df[78:(len(df)-1)].reset_index(drop=True)

#shift targets
dfamznShiftedTar = shiftTarget(dfamzn)
dfaaplShiftedTar = shiftTarget(dfaapl)
dfgoogShiftedTar = shiftTarget(dfgoog)

#split df into train/test
def trainTestSplit(dfStock):
    df = dfStock.copy()
    dfTrain = df[:4000]
    dfTest = df[4000:]
    return dfTrain, dfTest

dfamznTrain, dfamznTest = trainTestSplit(dfamznShiftedTar)
dfaaplTrain, dfaaplTest = trainTestSplit(dfaaplShiftedTar)
dfgoogTrain, dfgoogTest = trainTestSplit(dfgoogShiftedTar)

#combine train sets
def combTrainSet(*args):
    frms = [*args]
    combinedTrain = pd.concat(frms).reset_index(drop=True)
    return combinedTrain

combinedTrain = combTrainSet(dfamznTrain, dfaaplTrain, dfgoogTrain)

#find value to fill na sentiment values
def naFunc(combTrainSet):
    t_pos_neg_NAfill = ((combTrainSet['tPos']+combTrainSet['tNeg'])/2).min()
    t_neut_NAfill = combTrainSet['tNeut'].max()
    
    d_pos_neg_NAfill = ((combTrainSet['dPos']+combTrainSet['dNeg'])/2).min()
    d_neut_NAfill = combTrainSet['dNeut'].max()
    return t_pos_neg_NAfill, t_neut_NAfill, d_pos_neg_NAfill, d_neut_NAfill

t_pos_neg_NAfill, t_neut_NAfill, d_pos_neg_NAfill, d_neut_NAfill = naFunc(combinedTrain)

#fill Nan for missing sentiments - Train and Test
def fillNA(df):
    df = df.copy()
    df['tNeg'].fillna(t_pos_neg_NAfill, inplace=True)
    df['tNeut'].fillna(t_neut_NAfill, inplace=True)
    df['tPos'].fillna(t_pos_neg_NAfill, inplace=True)
    df['dNeg'].fillna(d_pos_neg_NAfill, inplace=True)
    df['dNeut'].fillna(d_neut_NAfill, inplace=True)
    df['dPos'].fillna(d_pos_neg_NAfill, inplace=True)
    return df

combinedTrainNoNa = fillNA(combinedTrain)

#build column transformer pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

def createPipeline(combinedTrainNoNa):
    dfTrain = combinedTrainNoNa.copy().drop(['large','small','up','down'], axis=1)
    dfTrainTargets = combinedTrainNoNa.copy()[['large','small','up','down']]
    
    sentiments = ['tNeg','tNeut','tPos','dNeg','dNeut','dPos']
    logReturns = ['close_open','high_open','low_open','spy_close_open',
                  'spy_high_open', 'spy_low_open','volPercent']
    weekday = ['weekday']
    artCnt = ['artCnt']
    vix = ['vixClose','spy_vix']
    
    minmaxer = MinMaxScaler((-1,1))
    
    full_pipeline = ColumnTransformer([
        ("sent", minmaxer, sentiments),
        ("artCnt", MinMaxScaler(), artCnt),
        ("ret", StandardScaler(), logReturns),
        ("weekday", OneHotEncoder(), weekday),
        ("pass", 'passthrough', vix)
        ])   

    transformedTrainSet = full_pipeline.fit_transform(dfTrain)
    transformedTrainSet = transformedTrainSet, dfTrainTargets

    return transformedTrainSet, full_pipeline
        
transformedTrainSet, full_pipeline = createPipeline(combinedTrainNoNa)

#prep and transform test sets
def prepTestData(dfTest):
    df = dfTest.copy()
    df = fillNA(df)
    dfNoTarget = df.copy().drop(['large','small','up','down'], axis=1)
    testTargets = df.copy()[['large','small','up','down']]
    transformedTestSet = full_pipeline.transform(dfNoTarget)      
    return transformedTestSet, testTargets

amznTransformedTestSet = prepTestData(dfamznTest)
aaplTransformedTestSet = prepTestData(dfaaplTest)
googTransformedTestSet = prepTestData(dfgoogTest)

#split dfs into 3Dim samples
from numpy import array, vstack
from sklearn.utils import shuffle

def split_sequences(sequences, targets, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		end_ix = i + n_steps
		if end_ix > len(sequences):
			break
		seq_x, seq_y = sequences[i:end_ix, :], targets[end_ix-1, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

n_steps = 5

#finish prep for RNN inputs and shuffle
def prepShuf(transformedTrainSet,amznTransformedTestSet,aaplTransformedTestSet,googTransformedTestSet,mag):
    if mag==True:
        x = 0
        y = 2
    else:
        x = 2
        y = 4
    amznX, amznY = split_sequences(transformedTrainSet[0][:4000], (array(transformedTrainSet[1][:4000]))[:,x:y], n_steps=n_steps)
    aaplX, aaplY = split_sequences(transformedTrainSet[0][4000:8000], (array(transformedTrainSet[1][4000:8000]))[:,x:y], n_steps=n_steps)
    googX, googY = split_sequences(transformedTrainSet[0][8000:], (array(transformedTrainSet[1][8000:]))[:,x:y], n_steps=n_steps)

    amznValX, amznValY = split_sequences(amznTransformedTestSet[0], (array(amznTransformedTestSet[1]))[:,x:y], n_steps=n_steps)
    aaplValX, aaplValY = split_sequences(aaplTransformedTestSet[0], (array(aaplTransformedTestSet[1]))[:,x:y], n_steps=n_steps)
    googValX, googValY = split_sequences(googTransformedTestSet[0], (array(googTransformedTestSet[1]))[:,x:y], n_steps=n_steps)
    
    splitCombTrain = vstack((amznX,aaplX,googX))
    splitCombTrainTargets = vstack((amznY,aaplY,googY))
    
    Xshuffled, Yshuffled = shuffle(splitCombTrain, splitCombTrainTargets, random_state=37)
    
    amznValShuffledX, amznValShuffledY = shuffle(amznValX, amznValY, random_state=41)
    aaplValShuffledX, aaplValShuffledY = shuffle(aaplValX, aaplValY, random_state=41)
    googValShuffledX, googValShuffledY = shuffle(googValX, googValY, random_state=41)
    
    return Xshuffled, Yshuffled, amznValShuffledX, amznValShuffledY, aaplValShuffledX, aaplValShuffledY, googValShuffledX, googValShuffledY

preppedMagData = prepShuf(transformedTrainSet,amznTransformedTestSet,aaplTransformedTestSet,googTransformedTestSet,mag=True)
preppedDirData = prepShuf(transformedTrainSet,amznTransformedTestSet,aaplTransformedTestSet,googTransformedTestSet,mag=False)

#functions to evaluate models and learning curves
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

def evalModel(model, DataSet=preppedMagData):
    u = model.predict(DataSet[6])
    v = np.argmax(u, axis = 1)
    w = np.argmax(np.array(DataSet[7]), axis=1)
    print(confusion_matrix(w,v))
    print("Accuracy:", accuracy_score(w,v))
    print("F1:", f1_score(w,v))
    print("Precision:", precision_score(w,v))
    print("Recall:", recall_score(w,v))
    print("AUC:", roc_auc_score(w,v))    

def plotRocCurve(model, DataSet=preppedMagData):
    v = model.predict(DataSet[6])[:,1]
    w = np.argmax(np.array(DataSet[7]), axis=1)
    fpr, tpr, thresholds = roc_curve(w, v)
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                  
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)   

def plotPrecRecall(model, DataSet=preppedMagData):
    v = model.predict(DataSet[6])[:,1]
    w = np.argmax(np.array(DataSet[7]), axis=1)
    precisions, recalls, thresholds = precision_recall_curve(w, v)
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

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

#build, compile, and train models
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

n_features = preppedMagData[0].shape[2]

np.random.seed(37)
tf.random.set_seed(37)

opt = Adam(1e-5, clipvalue=1.0)

#build LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(10, return_sequences=True, input_shape=[None, n_features]),
    tf.keras.layers.LSTM(10),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation='softmax')
])

#build GRU model
np.random.seed(37)
tf.random.set_seed(37)

model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(10, return_sequences=True, input_shape=[None, n_features]),
    tf.keras.layers.GRU(10),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation='softmax')
])

#build Conv1D model
np.random.seed(37)
tf.random.set_seed(37)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=20, kernel_size=2, strides=1, padding="valid", input_shape=[n_steps, n_features]),
    #tf.keras.layers.GRU(20, return_sequences=True),
    tf.keras.layers.GRU(20),
    tf.keras.layers.Dense(2, activation='softmax')
])

#build WaveNet model
np.random.seed(37)
tf.random.set_seed(37)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[n_steps, n_features]))
for rate in (1, 2, 4, 8) * 2:
    model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                  activation="relu", dilation_rate=rate))
model.add(tf.keras.layers.GRU(20, return_sequences=True))
model.add(tf.keras.layers.GRU(20))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

#compile and fit model
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
history = model.fit(x=preppedMagData[0], y=preppedMagData[1], epochs=40, validation_data=(preppedMagData[2],preppedMagData[3]), shuffle=(True), batch_size = 16)

plotLoss()
plotAcc()
evalModel(model)
plotRocCurve(model)
plotPrecRecall(model)

#Monte Carlo Dropout
tf.random.set_seed(42)
np.random.seed(42)

y_probs = np.stack([model(preppedMagData[6], training=True) for sample in range(100)])
y_probAve = y_probs.mean(axis=0)
y_std = y_probs.std(axis=0)

y_pred = np.argmax(y_probAve, axis=1)

#adjusting threshold
thresh=.70
z = model.predict(preppedMagData[6])
z1 = pd.DataFrame(z)
zz = np.array(preppedMagData[7])
z1['T'] = zz[:,0]
z1['F'] = zz[:,1]
z2 = z1.loc[(z1[0]>thresh) | (z1[1]>thresh)]
w = np.argmax(np.array(z2[['T','F']]), axis = 1)
v = np.argmax(np.array(z2[[0,1]]), axis = 1)

print(confusion_matrix(w,v))
print("Accuracy:", accuracy_score(w,v))
print("F1:", f1_score(w,v))
print("Precision:", precision_score(w,v))
print("Recall:", recall_score(w,v))
print("AUC:", roc_auc_score(w,v))    
    