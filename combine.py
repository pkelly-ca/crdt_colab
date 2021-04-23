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
date_current = "20210416"
gd_path = "/Users/User/Google Drive (patkellyatx@gmail.com)/CRDT/ribbon/"
df = pd.read_csv(gd_path + 'crdt_' + date_main + '.csv').fillna(-1)
df_new = pd.read_csv(gd_path + 'crdt_' + date_current + '.csv',header=[0,1])
#df_new.columns = df.columns.values
df_new.columns = ['_'.join(col) for col in df_new.columns.values]
display(df_new)
display(list(df_new)[0])
df_new = df_new.rename(columns={list(df_new)[0]:'State',list(df_new)[1]:'Date'})
df_new.columns = df_new.columns.str.replace('Hospitalizations','Hosp')
display(df)
display(df_new)
df = df_new.append(df)
display(df)
df = df.replace(-1,'')
df.to_csv('crdt_combined_'+date_current+'.csv',index=False)
