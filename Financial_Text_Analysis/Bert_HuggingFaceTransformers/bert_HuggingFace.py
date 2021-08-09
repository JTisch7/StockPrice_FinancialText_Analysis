# -*- coding: utf-8 -*-


from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import datetime 
import transformers
from transformers import BertTokenizer, TFBertModel

#get tokenizer and bert model from huggingFace
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_layer = TFBertModel.from_pretrained('bert-base-cased')

#read in and prepare Text dataset
dfText1 = pd.read_csv('data/dailyRedpolitics.csv', index_col=0)
dfText1= dfText1.drop([0]).reset_index(drop=True)
dfText1.afterEpoch = dfText1.afterEpoch.astype(int)
dfText1.beforeEpoch = dfText1.beforeEpoch.astype(int)
dfText1 = dfText1.replace(np.nan,'')
dfText1['sentence'] = dfText1.art0 + '. ' + dfText1.art1+ '. ' + dfText1.art2+ '. ' + dfText1.art3+ '. ' + dfText1.art4+ '. ' + dfText1.art5 + '. ' + dfText1.art6+ '. ' + dfText1.art7+ '. ' + dfText1.art8+ '. ' + dfText1.art9+ '. ' + dfText1.art10
dfText1['sentence'] = dfText1['sentence'].astype('string')

#read in and prepare Spx dataset
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

dfText1['up'] = dfSpx['up']
dfText1['down'] = dfSpx['down']

#set max length of sentences to be tokenized
MAX_SEQ_LEN=100

#create lists for tokenized inputs
def getTokens():
    input_idst = []
    attention_maskst = []
    
    for sentence in dfText1['sentence']:
        encoded_new = tokenizer.encode_plus(
                                            sentence,                      
                                            add_special_tokens = True,  
                                            padding = True,
                                            max_length = MAX_SEQ_LEN,             
                                            truncation=True,
                                            return_attention_mask = True,     
                                            #return_tensors = 'tf'        
                                            )
        input_idst.append(encoded_new['input_ids'])
        attention_maskst.append(encoded_new['attention_mask'])
        
    return input_idst, attention_maskst
    
input_idst, attention_maskst = getTokens()

#reshape inputs
input_idst = np.reshape(input_idst,(2780,-1))
attention_maskst = np.reshape(attention_maskst,(2780,-1))

#build model
input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')
sequence_output = bert_layer([input_ids, input_masks_ids])["last_hidden_state"]
x = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(sequence_output)
x = tf.keras.layers.Dense(768, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(10, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.BatchNormalization()(x)
out = tf.keras.layers.Dense(2,activation="softmax")(x)

model = tf.keras.Model(inputs=(input_ids, input_masks_ids), outputs=out)

#freeze bert_layer (can be partially or fully unfrozen and fine-tuned after initial training)
for layer in model.layers[2:3]:
    layer.trainable=False
    print(layer, layer.trainable)

#compile, fit, summarize, and plot model
model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
history = model.fit([input_idst,attention_maskst],dfText1[['up','down']],epochs=1 ,validation_split=0.2, batch_size=32,shuffle=True)
tf.keras.utils.plot_model(model, show_shapes=True)
model.summary()

#look at/modify layers in model and/or bert_layer for fine-tuning, then compile and train more
for layer in model.layers[2:3]:
    #layer.trainable=True
    print(layer, layer.trainable)
        
for layer in bert_layer.layers[:]:
    if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
       for layer in layer.encoder.layer[:]:
            #layer.trainable = True
            print(layer, layer.trainable)

for layer in bert_layer.layers[:]:
    if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
       for idx, layer in enumerate(layer.encoder.layer):
           if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
              #layer.trainable = False
              print(layer, layer.trainable)


#save and load model
#model.save("testSave")
#model2 = tf.keras.models.load_model("testSave")
#model2.summary()

