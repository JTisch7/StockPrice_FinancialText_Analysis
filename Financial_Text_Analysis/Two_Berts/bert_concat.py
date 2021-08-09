# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:58:02 2020

@author: Jonathan
"""


from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as hub
import bert
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import datetime 
import matplotlib.pyplot as plt

#get bert layers
bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=False)
bert_layer2=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=False)

#set max length of text sequence
MAX_SEQ_LEN=100

#get tokenizer set up for both bert layers
FullTokenizer=bert.bert_tokenization.FullTokenizer

vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
tokenizer1=FullTokenizer(vocab_file,do_lower_case)

vocab_file2=bert_layer2.resolved_object.vocab_file.asset_path.numpy()
do_lower_case2=bert_layer2.resolved_object.do_lower_case.numpy()
tokenizer2=FullTokenizer(vocab_file2,do_lower_case2)

#functions for bert sentence preprocessing/tokenizing
def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, tokenizer, max_seq_length):
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def create_single_input(sentence,MAX_LEN,tokenizer):
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:MAX_LEN]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
 
    ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids,masks,segments

def create_input_array(sentences, tokenizer):
    input_ids, input_masks, input_segments = [], [], []
    
    for sentence in tqdm(sentences, position=0, leave=True):
        ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2,tokenizer)
        
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]

#read in and prepare Text dataset #1
dfText1 = pd.read_csv('data/dailyRedpolitics.csv', index_col=0)
dfText1= dfText1.drop([0]).reset_index(drop=True)
dfText1.afterEpoch = dfText1.afterEpoch.astype(int)
dfText1.beforeEpoch = dfText1.beforeEpoch.astype(int)
dfText1 = dfText1.replace(np.nan,'')
dfText1['sentence'] = dfText1.art0 + '. ' + dfText1.art1+ '. ' + dfText1.art2+ '. ' + dfText1.art3+ '. ' + dfText1.art4+ '. ' + dfText1.art5 + '. ' + dfText1.art6+ '. ' + dfText1.art7+ '. ' + dfText1.art8+ '. ' + dfText1.art9+ '. ' + dfText1.art10
dfText1['sentence'] = dfText1['sentence'].astype('string')

#read in and prepare Text dataset #2
dfText2 = pd.read_csv('data/dailyRednews.csv', index_col=0)
dfText2= dfText2.drop([0]).reset_index(drop=True)
dfText2.afterEpoch = dfText2.afterEpoch.astype(int)
dfText2.beforeEpoch = dfText2.beforeEpoch.astype(int)
dfText2 = dfText2.replace(np.nan,'')
dfText2['sentence2'] = dfText2.art0 + '. ' + dfText2.art1+ '. ' + dfText2.art2+ '. ' + dfText2.art3+ '. ' + dfText2.art4+ '. ' + dfText2.art5 + '. ' + dfText2.art6+ '. ' + dfText2.art7+ '. ' + dfText2.art8+ '. ' + dfText2.art9+ '. ' + dfText2.art10
dfText2['sentence2'] = dfText2['sentence2'].astype('string')

#read in and prepare stock price dataset
dfSpx = pd.read_csv('data/spx10yr.csv')
dfSpx = dfSpx.iloc[4159:].reset_index(drop=True)
dfSpx['date1'] = pd.to_datetime(dfSpx['date'])
dfSpx['startDate'] = (dfSpx['date1'] - datetime.datetime(1970,1,1)).dt.total_seconds()
dfSpx.startDate = dfSpx.startDate.astype(int)
min30 = 1800
dfSpx['startDate'] = dfSpx['startDate'] + (27*min30)
dfSpx['endDate'] = dfSpx['startDate'] + (13*min30)
dfSpx['up'] = 0
dfSpx['down'] = 0
dfSpx['change'] = dfSpx.close-dfSpx.open
dfSpx.loc[(dfSpx.change >= 0), 'up'] = 1
dfSpx.loc[(dfSpx.change < 0), 'down'] = 1

#add features and targ to dataset
dfText1['sentence2'] = dfText2['sentence2']
dfText1['up'] = dfSpx['up']
dfText1['down'] = dfSpx['down']

#split into train/test set
dfText1 = dfText1.sample(frac=1)
X_train, X_test, y_train, y_test = train_test_split(dfText1[['sentence','sentence2']], dfText1[['up', 'down']], test_size=0.20, random_state=42)

#inputs to 1st and 2nd bert_layer
inputs = dict(
    input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_word_ids"),
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_mask"),
    input_type_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids"))
inputs2 = dict(
    input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_word_ids2"),
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_mask2"),
    input_type_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids2"))

#1st input model
sequence_output = bert_layer(inputs)["sequence_output"]
x = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(sequence_output)
x = tf.keras.layers.Dense(768, activation="relu")(x)
#2nd input model
sequence_output2 = bert_layer2(inputs2)["sequence_output"]
x2 = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(sequence_output2)
x2 = tf.keras.layers.Dense(768, activation="relu")(x2)
#concat 1st and 2nd
x = tf.keras.layers.concatenate([x, x2])
x = tf.keras.layers.Dense(10, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(10, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
out = tf.keras.layers.Dense(2,activation="softmax")(x)

#compile model
model = tf.keras.Model(inputs=[inputs, inputs2], outputs=out)
model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

#tokenize and combine inputs
Inputs=create_input_array(X_train['sentence'], tokenizer1)
Inputs2=create_input_array(X_train['sentence2'], tokenizer2)
inputsCombined = Inputs+Inputs2

#fit, summarize, and plot model
history = model.fit(inputsCombined,y_train,epochs=1,batch_size=32,validation_split=0.2,shuffle=True)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

#unfreeze bert model layers
for layer in model.layers[:]:
    print(layer,layer.trainable)

for layer in model.layers[6:8]:
    layer.trainable = True
    
for layer in model.layers[:]:
    print(layer,layer.trainable)
    
#look at layers within one bert model
for layer in model.layers[6:7]:
    for var in layer.variables[:]:
        print(var.name)
        
#compile and fit again with lowered learning rate
model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-6), metrics=['accuracy'])
history = model.fit(inputsCombined,y_train,epochs=1,batch_size=32,validation_split=0.2,shuffle=True)
    
#evaluate model
def evalModel():
    test_inputs=create_input_array(X_test['sentence'],tokenizer1)
    test_inputs2=create_input_array(X_test['sentence2'],tokenizer2)
    test_inputsCombined = test_inputs+test_inputs2
    u = model.predict(test_inputsCombined)
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

#save and load model
model.save("testSave")
model2 = tf.keras.models.load_model("testSave")
model2.summary()

