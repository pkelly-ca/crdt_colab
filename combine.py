# Add latest ribbon to dataset
# Necessary Libraries
from IPython.display import Markdown, display
import pandas as pd
import requests
import numpy as np
from io import StringIO, BytesIO
import getopt,sys
import datetime
from datetime import date, timedelta
import os, time
os.environ['TZ'] = 'America/New_York'
time.tzset()

date_main = "20210307"
date_start = "20210416"
date_finish = date.today()
#date_finish = date_finish - timedelta(1)
date_finish = date_finish.strftime('%Y%m%d')

#date_finish = "20210504"

gd_path = "/Users/User/Google Drive (patkellyatx@gmail.com)/CRDT/ribbon/"
df = pd.read_csv(gd_path + 'crdt_' + date_main + '.csv').fillna(-1)
display(df)

d_start = datetime.datetime.strptime(date_start, '%Y%m%d')
d_end = datetime.datetime.strptime(date_finish, '%Y%m%d')
d_count = int((d_end-d_start).days)
for d in (d_start + timedelta(n) for n in range(0,d_count+1)):
  print(d.strftime("%Y%m%d"))
  df_new = pd.read_csv(gd_path + 'crdt_' + d.strftime("%Y%m%d") + '.csv',header=[0,1])
  df_new.columns = ['_'.join(col) for col in df_new.columns.values]
  display(df_new)
  display(list(df_new)[0])
  df_new = df_new.rename(columns={list(df_new)[0]:'State',list(df_new)[1]:'Date'})
  df_new.columns = df_new.columns.str.replace('Hospitalizations','Hosp')
  df = df_new.append(df)

#df = df.replace(-1,'')
df.replace(-1,'').to_csv(gd_path + 'crdt_combined_'+date_finish+'.csv',index=False)
df = df.set_index(['Date','State'])
df_change = df.pct_change(periods=-45) * 100
display(df_change)
df_change.to_csv(gd_path + 'crdt_change_'+date_finish+'.csv')
df_neg = df_change.loc[int(date_finish)].copy()
df_neg[df_neg >= 0] = ''
df_neg.to_csv('neg.csv')
display(df_neg)
df_check = df_change.loc[int(date_finish)].abs()
df_check[df_check < 1] = ''
display(df_check)
df_check.to_csv('check.csv')
df = df_check.replace(r'^\s*$', np.nan, regex=True)
s_stack = df.stack()
s_stack.to_csv('poslist.csv')
print(s_stack)
df = df_neg.replace(r'^\s*$', np.nan, regex=True)
s_stack = df.stack()
s_stack.to_csv('neglist.csv')

