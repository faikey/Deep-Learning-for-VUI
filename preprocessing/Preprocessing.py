
# coding: utf-8
import pandas as pd
import csv
import os
import re
import math

fullResponse = {}
responseDf = pd.DataFrame(columns=['response','name','label'])
filepath = ""
txt = glob.glob(filepath)
count = 0
label = 1
for filename in sorted(txt):
    #print(filename)
    df = pd.read_csv(filename,encoding="iso-8859-1" )
    for index, row in df.iterrows():
        if isinstance(row['response'], str):
            if row['name'] not in fullResponse:
                fullResponse[row['name']] = row['response']
            else:
                fullResponse[row['name']] = fullResponse[row['name']] + " " + row['response']
    new_df = pd.DataFrame.from_dict(fullResponse, orient='index')
    new_df['name'] = new_df.index
    new_df=new_df.reset_index()
    new_df['label'] = label
    del new_df['index']
    new_df.columns = ['response','name','label']
    responseDf = pd.concat([responseDf, new_df])
    count += 1
    if count%2==0:
        label += 1
        fullResponse = {}


df_full = df[['response','label']]


from sklearn.model_selection import train_test_split
train, test = train_test_split(df_full, test_size=0.2, random_state=42)


train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)

