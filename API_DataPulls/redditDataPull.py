# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:57:14 2020

@author: Jonathan
"""

import math
import json
import requests
import numpy as np
import time
import pandas as pd

start_time = time.time()

df = pd.read_csv('spy_30Min2yrsNum2.csv')

title = [[]]*20
doc_count = [[]]*20
beforelist = []
afterlist = []
df2 = pd.DataFrame()
    
def getPushshiftData(after, before, subred):
    def fire_away(after=after, before=before, subred=subred):
        url = 'https://api.pushshift.io/reddit/search/comment/?subreddit='+str(subred)+'&after='+str(after)+'&before='+str(before)+'&sort=desc&aggs=link_id'
        response = requests.get(url)
        assert response.status_code == 200
        data = json.loads(response.content)
        return data
    current_tries = 1
    while current_tries < 5:
        try:
            time.sleep(.3)
            response = fire_away()
            return response
        except:
            time.sleep(.3)
            current_tries += 1
    return fire_away()


def loopDayData(after, before, numPerDay, subred):
    tries = 1
    while tries < 5:
        data = getPushshiftData(after = after, before = before, subred = subred)
        if type(data['aggs']['link_id']) is not list:
            tries += 1
            time.sleep(10)
        else:
            break    
    beforelist.append(before)
    afterlist.append(after)
    l=0
    while l < numPerDay:
        if l < len(data['aggs']['link_id']): 
            print('l', l)
            title[l].append(data['aggs']['link_id'][l]['data']['title'])
            doc_count[l].append(data['aggs']['link_id'][l]['doc_count'])
            l+=1
        else:
            print('l', l)
            title[l].append('nan')
            doc_count[l].append(0)
            l+=1
            

def multiDayLoop(numPerDay, subred):
    y=0
    while y < (len(df['epochDate'])-1):
        print('y' , y)
        after = ((df['epochDate'][y])/1000)
        after = math.trunc(after)
        after = str(after)
        before = ((df['epochDate'][y+1])/1000)
        before = math.trunc(before)
        before = str(before)
        loopDayData(after = after, before = before, numPerDay=numPerDay, subred=subred)
        time.sleep(.3)
        y+=1
        
        
def loopAndCreateDF(numPerDay, subred):
    global df2
    multiDayLoop(numPerDay=numPerDay, subred=subred)
    df2['afterEpoch'] = afterlist
    df2['after'] = pd.to_datetime(df2['afterEpoch'], unit='s').dt.tz_localize('utc').dt.tz_convert('America/Los_Angeles')
    df2['beforeEpoch'] = beforelist
    df2['before'] = pd.to_datetime(df2['beforeEpoch'], unit='s').dt.tz_localize('utc').dt.tz_convert('America/Los_Angeles')
    e = 0
    while e < numPerDay:
        df2['art'+str(e)] = title[e]
        df2['cnt'+str(e)] = doc_count[e]
        e += 1
    df2.loc[len(df)] = 0
    df2 = df2.shift(1)
    df2.to_csv('red'+str(subred)+'.csv')


loopAndCreateDF(20, 'politics')

    
print("--- %s seconds ---" % (time.time() - start_time))
