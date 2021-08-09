# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:59:14 2020

@author: Jonathan
"""

# Package used to execute HTTP POST request to the API
import json
import urllib.request
import pandas as pd
import time
import pytz
from dateutil.parser import parse

#format request
def newsFilterPull(frm, ticker, company, begin, end):
    # API key
    API_KEY = 'API_KEY'

    # Define the filter parameters
    queryString = "(title:{company} OR title:{ticker}) AND publishedAt:[{begin} TO {end}]".format(company=company,ticker=ticker,begin=begin,end=end)
    payload = {
        "type": "filterArticles",
        "queryString": queryString,
        "from": frm,
        "size": 50
        }
    
    #establish and send request
    jsondata = json.dumps(payload)
    jsondataasbytes = jsondata.encode('utf-8')
    
    API_ENDPOINT = "https://api.newsfilter.io/public/actions?token={}".format(API_KEY)
    req = urllib.request.Request(API_ENDPOINT)

    req.add_header('Content-Type', 'application/json; charset=utf-8')
    req.add_header('Content-Length', len(jsondataasbytes))

    response = urllib.request.urlopen(req, jsondataasbytes)
    res_body = response.read()

    articles = json.loads(res_body.decode("utf-8"))
    return articles

#looping and recording function
def createDateframe(frm, ticker, company, begin, end):
    start_time = time.time()
    date = []
    title = []
    desc = []
    source = []

    #create looping function
    def loopingFunc(frm, ticker, company, begin, end):
        y = frm + 1
        #loop through articles pulling and recording 50 at a time
        while frm < y:
            time.sleep(2)
            print('running - slowly but surely : ', frm)
            articles = newsFilterPull(frm=frm, ticker=ticker, company=company, begin=begin, end=end)
            l = 0
            #record title, date, source, and description if it exists
            while l < len(articles['articles']):
                title.append(articles['articles'][l]['title'])
                date.append(articles['articles'][l]['publishedAt'])
                source.append(articles['articles'][l]['source']['name'])
                try:
                    desc.append(articles['articles'][l]['description'])
                except:
                    desc.append('')
                l+=1
            y = articles['total']['value']
            frm += 50
            #when at the end and if total exeeds 10,000 restart request with updated parameters
            if (frm >= y) and (articles['total']['relation'] == 'gte'):
                timestampStr = parse(date[-1]).strftime("%Y-%m-%d")
                loopingFunc(frm=0, ticker=ticker, company=company, begin=begin, end=timestampStr)
            
    #run looping func
    loopingFunc(frm=frm,ticker=ticker,company=company,begin=begin,end=end)
       
    #create dataframe and copy lists to dataframe
    df = pd.DataFrame()        
    df['date'] = date
    df['title'] = title
    df['description'] = desc
    df['source'] = source
    
    #convert dates to same timezone
    for j in range(len(df)):
        df['date'][j] = parse(df['date'][j]).astimezone(pytz.timezone('America/Los_Angeles'))
     
    end_time = time.time()
    totalTime = (end_time-start_time)/60
    #df.to_csv('news'+str(ticker)+'2yrs.csv')
    return df, totalTime
    
    
df, totalTime = createDateframe(frm=0, ticker='GS', company='Goldman Sachs', begin='2019-03-08', end='2021-03-13')


