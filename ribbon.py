# Create summary spreadsheet
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

def colstr2int(df,col):
  df.loc[:,col] = df.loc[:,col].replace(',','', regex=True)
  df[col] = df[col].astype('int')

def init_ribbon():
  headings = ['Total','White','Black','Hispanic','Asian','AIAN','NHPI','Multiple','Other','Unknown','Hispanic','Non-Hispanic','Unknown']
  headings = headings*4
  types = ['Cases','Deaths','Tests','Hospitalizations']
  types = np.repeat(types,13)
  df = pd.DataFrame([-1]*52,columns=[state],index=[types,headings],dtype=np.int)
#  df = pd.DataFrame(np.zeros((52,1)),columns=[state],index=[types,headings],dtype=np.int)
  return df

def st_csv(i,path,date,state):
  fn = path + "/" + state + "-" + date + '-' + str(i) + '.csv'
  return pd.read_csv(fn)

def st_map(df_st,df,map):
  i = 0
  for index in map[0]:
    if index < 13:
      col = 'Cases'
    elif index < 26:
      col = 'Deaths'
    elif index < 39:
      col = 'Tests'
    else:
      col = 'Hospitalizations'
    df_st.iloc[index,0]=df.loc[map[1][i],col]
    i+=1
  return df_st

def load_state_keys(csv):
  df = pd.read_csv(csv,index_col=0,header=[0,1])
  return df

def get_key(keys):
  map = [[],[]]
  i = 0
  for val in keys:
    if val != -1:
      map[0].append(i)
      map[1].append(val)
    i+=1
  return map

def state_common(df,keys,state):
  display(df)
  # Init Ribbon
  df_st = init_ribbon()
  # Baseline Mapping
  mapping = get_key(keys.loc[state])
  df_st = st_map(df_st,df,mapping)
  return df_st


def runAK(path,date,state,keys):
  # Read state file(s)
  df = st_csv(1,path,date,state)
  # State file prep
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  df_st.iloc[48]=df.iloc[11,2]+df.iloc[12,2]
  df_st.iloc[51]=df.iloc[2,2]+df.iloc[3,2]
  df_st.iloc[22]=df.iloc[11,3]+df.iloc[12,3]
  df_st.iloc[25]=df.iloc[2,3]+df.iloc[3,3]
  # Return ribbon
  return df_st


def runAL(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,5):
    df[i] = st_csv(i,path,date,state)
  df_eth = df[1].merge(df[3])
  df_eth = df_eth.drop(['Unnamed: 0','EthnicityCat'],axis=1)
  df_eth.columns = ['Category','Cases','Deaths']
  df_race = df[2].merge(df[4])
  df_race = df_race.drop(['Unnamed: 0'],axis=1)
  df_race.columns = ['Category','Cases','Deaths']
  df = df_race.append(df_eth)
  df = df.reset_index(drop=True)
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runAR(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,3):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].T
    df[i]=df[i].reset_index()
    print('i =',i)
  df = df[1].join(df[2],lsuffix='_l',rsuffix='_r')
  df.columns = ['','Cases','','Deaths']
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runCA(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,2):
    df[i] = st_csv(i,path,date,state)
  df = df[1]
  df.columns = ['','Category','Cases','Deaths','']
  non_h_cases = df['Cases'].sum() - df['Cases'].iloc[7] - df['Cases'].iloc[3]
  non_h_deaths = df['Deaths'].sum() - df['Deaths'].iloc[7] - df['Deaths'].iloc[3]
  df.loc[len(df.index)] = [0,'Non-Hispanic',non_h_cases,non_h_deaths,'2021-03-31']
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runCO(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,4):
    df[i] = st_csv(i,path,date,state)
  # Pre-processing
  df_tots = df[1]
  df = df[2].join(df[3]['Death Count']).drop(['value','category','index','Unnamed: 0'],axis=1)
  df.loc[len(df.index)] = ['Totals',df_tots['value'].loc[0],df_tots['value'].loc[5]]
  df.columns = ['metric','Cases','Deaths']
  df['Cases']=df['Cases'].astype('int')
  df['Deaths']=df['Deaths'].astype('int')
  non_h_cases = df['Cases'].sum() - df['Cases'].iloc[9] - df['Cases'].iloc[3] - df['Cases'].iloc[7] 
  non_h_deaths = df['Deaths'].sum() - df['Deaths'].iloc[9] - df['Deaths'].iloc[3] - df['Deaths'].iloc[7] 
  df.loc[len(df.index)] = ['Non-Hispanic',non_h_cases,non_h_deaths]
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runCT(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,2+1):
    df[i] = st_csv(i,path,date,state)
  df_tots = df[1]
  df = df[2].drop('Unnamed: 0',axis=1)
  df.columns = ['Category','Cases','Deaths']
  non_h_cases = df['Cases'].sum() - df['Cases'].iloc[0] - df['Cases'].iloc[7]
  non_h_deaths = df['Deaths'].sum() - df['Deaths'].iloc[0] - df['Deaths'].iloc[7]
  df.loc[len(df.index)] = ['Totals',df_tots['Total cases'].loc[0],df_tots['Total deaths'].loc[0]]
  df.loc[len(df.index)] = ['Non-Hispanic',non_h_cases,non_h_deaths]
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runDC(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,2+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  df = df[1].merge(df[2],on='Race',how='outer').fillna(0)
  df.columns = ['Category','Cases','Deaths']
  df['Cases'] = df['Cases'].astype('int')
  df['Deaths'] = df['Deaths'].astype('int')
  df.loc[len(df.index)] = ['Unk+Ref (Race)',df['Cases'].loc[1]+df['Cases'].loc[9],0]
  df.loc[len(df.index)] = ['Unk+Ref (Eth)',df['Cases'].loc[2]+df['Cases'].loc[10],0]
  non_h_deaths = df['Deaths'].sum() - df['Deaths'].iloc[0] - df['Deaths'].iloc[1] - df['Deaths'].iloc[2] 
  df.loc[len(df.index)] = ['Non-Hispanic Deaths',0,non_h_deaths]
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runDE(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,4+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  # Pre-processing
  df[2].columns = ['Category','Cases']
  df[3].columns = ['Category','Deaths']
  df[4].columns = ['Category','Tests']
  df_tots = df[1]
  df = df[2].merge(df[3],on='Category').merge(df[4],on='Category')
  non_h_cases = df['Cases'].sum() - df['Cases'].iloc[2] - df['Cases'].iloc[5]
  non_h_deaths = df['Deaths'].sum() - df['Deaths'].iloc[2] - df['Deaths'].iloc[5]
  non_h_tests = df['Tests'].sum() - df['Tests'].iloc[2] - df['Tests'].iloc[5]
  df.loc[len(df.index)] = ['Non-Hispanic',non_h_cases,non_h_deaths,non_h_tests]
  df.loc[len(df.index)] = ['Total',df_tots.loc[0,'Total'],df_tots.loc[1,'Total'],df_tots.loc[2,'Total']]
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runFL(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,3+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
  hosp_total = df[1]['Values'].sum()
  df_tots = df[2]
  colstr2int(df_tots,'Value')
  case_total = df_tots[df_tots['Metric']=='Total cases'].loc[0,'Value']
  deaths_total = df_tots.loc[df_tots.Metric=='Florida resident deaths','Value'].iloc[0] + df_tots.loc[df_tots.Metric=='Non-Florida resident deaths','Value'].iloc[0]
  non_res = ['Non-Florida residents',df_tots.iloc[4,1],df[1].loc[1,'Values'],df_tots.iloc[24,1]]
  df = df[3].drop(['Unnamed: 3','Unnamed: 5','Unnamed: 7'],axis=1)
  for cat in ['Cases', 'Hospitalizations', 'Deaths']:
    colstr2int(df,cat)
  df.loc[len(df.index)] = non_res
  df.loc[len(df.index)] = ['Totals',case_total,hosp_total,deaths_total]
  df = df.set_index('Race and ethnicity')
  for i in range(0,3):
    df.iloc[i*4] -= df.iloc[i*4+1]
  df = df.groupby(['Race and ethnicity'],sort=False).sum()
  df.iloc[[3,6]] += df.iloc[7]
  df = df.reset_index()
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runGA(path,date,state,keys):
  # Read state file(s)
  df = {}
  for i in range(1,2+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  df_tots = df[2]
  df = df[1]
  df.columns = ['Ethnicity','Race','Cases','Hospitalizations','Deaths']
  df = df.set_index(['Ethnicity','Race'])
  display(df)
  df.loc[('Unknown','Unknown'),:] += [df_tots.loc[0,'antigen_cases'],0,df_tots.loc[0,'probable_deaths']]
  display(df)
  s_tot = df.sum()
  df_tot = pd.DataFrame([s_tot.values],columns=s_tot.index,index=['Totals']).reset_index().rename({'index':'Category'},axis=1)
  df_eth = df.sum(level=0).reset_index().rename({'Ethnicity':'Category'},axis=1)
  df_race = df.loc[['Non-Hispanic/ Latino','Unknown']].sum(level=1).reset_index().rename({'Race':'Category'},axis=1)
  df = pd.concat([df_eth, df_race, df_tot]).reset_index(drop=True)
#  df.loc['Total', :] = df.sum().values
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runGU(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  # Pre-processing
  df = df[1].merge(df[2],on='Category')
  df.loc[len(df.index)] = ['Totals',df.Cases.sum(),df.Deaths.sum()]
  df.loc[len(df.index)] = ['Asian',df.Cases.loc[[3,5]].sum(),df.Deaths.loc[[3,5]].sum()]
  df.loc[len(df.index)] = ['NHPI',df.Cases.loc[[0,1,6]].sum(),df.Deaths.loc[[0,1,6]].sum()]
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runHI(path,date,state,keys):
  # Read state file(s)
  num_files = 4 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  df_tots = df
  df = df[4].drop(['Cases','State Population','State Population.1'],axis=1)
  df.columns=['Race','Cases','Deaths','Hospitalizations']
  df.loc[len(df.index)] = ['Totals',df_tots[1].loc[0,'Total Cases'],df_tots[2].loc[0,'Deaths'],df_tots[3].loc[0,'Hospital']]
  df.loc[len(df.index)] = pd.Series(['Unknowns'],index=['Race']).append(df.iloc[10,[1,2,3]] - df.iloc[9,[1,2,3]])
  df.loc[len(df.index)] = ['Asian',df.Cases.loc[[3,4,5,6]].sum(),df.Deaths.loc[[3,4,5,6]].sum(),df.Hospitalizations.loc[[3,4,5,6]].sum()]
  df.loc[len(df.index)] = ['NHPI',df.Cases.loc[[1,2]].sum(),df.Deaths.loc[[1,2]].sum(),df.Hospitalizations.loc[[1,2]].sum()]
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runID(path,date,state,keys):
  # Read state file(s)
  num_files = 7 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  df_tot_cases = df[1]
  df_tot_deaths = df[2]
  df[4].loc[len(df[4].index)] = ['Known Race Total',df[4].Cases.sum()]
  df[5].loc[len(df[5].index)] = ['Known Eth Total',df[5].Cases.sum()]
  df[6].loc[len(df[6].index)] = ['Known Race Total',df[6].Deaths.sum()]
  df[7].loc[len(df[7].index)] = ['Known Eth Total',df[7].Deaths.sum()]
  df_cases = df[4].append(df[5])
  df_deaths = df[6].append(df[7])
  df_deaths.loc[:,'Category'] = df_deaths.loc[:,'Category'].replace('/',' or ', regex=True)
  df_deaths.loc[:,'Category'] = df_deaths.loc[:,'Category'].replace('n-','t ', regex=True)
  df = df_cases.merge(df_deaths,how='outer',on='Category').fillna(-1)
  df.loc[len(df.index)] = ['Total',df_tot_cases.Value[0],df_tot_deaths.Value[0]]
  df.loc[len(df.index)] = ['Race Unknown',df.Cases[10]-df.Cases[6],df.Deaths[10]-df.Deaths[6]]
  df.loc[len(df.index)] = ['Eth Unknown',df.Cases[10]-df.Cases[9],df.Deaths[10]-df.Deaths[9]]
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runIL(path,date,state,keys):
  # Read state file(s)
  num_files = 1 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  df = df[1]
  df.index = df['Category']
  df = df.drop('Category',axis=1)
  df.loc['Total']=df.sum()
  df.loc['Non-Hispanic']=df.loc['Total']-df.loc['Left Blank']-df.loc['Hispanic']
  df = df.reset_index()
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runIN(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    df[i]=df[i].loc[:,~df[i].columns.str.contains('PCT')]
    df[i].columns = ['Category','Tests','Cases','Deaths']
    df[i].index = df[i]['Category']
    df[i] = df[i].drop('Category',axis=1)
    df[i].loc['Total']=df[i].sum()
    display(df[i])
  df = df[1].append(df[2]).reset_index()
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runKY(path,date,state,keys):
  # Read state file(s)
  num_files = 3 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    if i == 3:
      df[i]=df[i].drop('Unnamed: 0.1',axis=1)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    if i < 3:
      df[i].columns = ['Cases','Category','Deaths']
      df[i].index = df[i]['Category']
      df[i] = df[i].drop('Category',axis=1)
      df[i].loc['Known total']=df[i].sum()
    display(df[i])
  # Pre-processing
  df[1].loc['Total'] = [df[3].loc[0,'Total'],df[3].loc[1,'Total']]
  df[1].loc['Unknown'] = df[1].loc['Total'] - df[1].loc['Known total']
  df[2].loc['Total'] = [df[3].loc[0,'Total'],df[3].loc[1,'Total']]
  df[2].loc['Unknown'] = df[2].loc['Total'] - df[2].loc['Known total']
  df = df[1].append(df[2]).reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runLA(path,date,state,keys):
  # Read state file(s)
  num_files = 1 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop('Unnamed: 0',axis=1)
    display(df[i])
  df = df[1].set_index(['LDH_Region','Race'])
  df = df.sum(level=1)
  df.loc['Total']=df.sum()
  df = df.reset_index()
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runMA(path,date,state,keys):
  # Read state file(s)
  num_files = 4 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    if i < 4:
      df[i]=df[i].drop(['Unnamed: 0','Date'],axis=1)
    else:
      df[i]=df[i].drop(['Date'],axis=1)
    display(df[i])
  case_tots = df[1]
  death_tots = df[2]
  known_tots = df[4]
  df = df[3].iloc[:,[0,1,2,3]]
  df.columns = ['Category','Cases','Hospitalizations','Deaths']
  df = df.set_index(['Category'],drop=True)
  df.loc['Known Total'] = df.sum()
  df.loc['Non Hispanic'] = df.loc['Known Total']-df.loc['Hispanic']-df.loc['Unknown, missing, or refused']
  df.loc['Total'] = [case_tots.loc[0,'Positive Total']+case_tots.loc[0,'Probable Total'],
                     known_tots.loc[0,'Ever Hospitaltized'],
                     death_tots.loc[0,'DeathsConfTotal']+death_tots.loc[0,'DeathsProbTotal']]
  df.loc['Unknown'] = df.loc['Total']-df.loc['Known Total']+df.loc['Unknown, missing, or refused']
  df = df.reset_index()
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runMD(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  tots = df[2]
  df = df[1]
  df['Deaths']=df['Deaths'].str.replace(r"[()]","")
  colstr2int(df,'Deaths')
  df['Unnamed: 4']=df['Unnamed: 4'].str.replace(r"\*","")
  df.columns = ['Category','Cases','Confirmed','Probable']
  colstr2int(df,'Probable')
  df['Deaths'] = df['Confirmed']+df['Probable']
  df=df.drop(['Confirmed','Probable'],axis=1)
  df = df.set_index(['Category'],drop=True)
  df.loc['Total'] = df.sum()
  df.loc['Non Hispanic'] = df.loc['Total']-df.loc['Hispanic']-df.loc['Data not available']
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runME(path,date,state,keys):
  # Read state file(s)
  num_files = 1 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    display(df[i])
  df = df[1].loc[:,['Race / Ethnicity Dimension','Cases','Hospitalizations','Deaths']]
  df = df.set_index(['Race / Ethnicity Dimension'],drop=True)
  df.loc['Total'] = df.sum()
  df = df.reset_index()
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runMI(path,date,state,keys):
  # Read state file(s)
  num_files = 1 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  df = df[1].fillna(0)
  df['Cases'] = df['Confirmed Cases']+df['Probable Cases']
  df['Deaths'] = df['Confirmed Deaths']+df['Probable Deaths']
  df = df.loc[:,['Category','Cases','Deaths']]
  df = df.set_index(['Category'],drop=True)
  df.loc['Total'] = df.sum()
  df = df.reset_index()
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runMN(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    display(df[i])
  # Pre-processing
  df[1]=df[1].drop(['Unnamed: 0'],axis=1)
  df[2]=df[2].rename({'Unnamed: 0':'index'},axis=1)
  df = df[1].merge(df[2],how='outer',on='index')
  df.columns = ['Category','Cases','Deaths']
  df.loc[len(df.index)]=['NH',df.Cases[1]-df.Cases[9]-df.Cases[10],df.Deaths[11]-df.Deaths[19]-df.Deaths[20]]
  df['Hospitalizations']=df['Deaths']
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runMO(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0','Antigen Tests','Serology Tests'],axis=1)
    display(df[i])
  # Pre-processing
  df[1] = df[1].rename({'Race':'Category'},axis=1)
  df[2] = df[2].rename({'Ethnicity':'Category'},axis=1)
  df = df[1].append(df[2]).reset_index(drop=True)
  df = df.rename({'Confirmed Deaths':'Deaths','PCR Tests':'Tests'},axis=1)
  df['Cases']=df['Confirmed Cases']+df['Probable Cases']
  df = df.drop(['Confirmed Cases','Probable Cases'],axis=1)
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runMS(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df[1] = df[1].rename({'Tot Cases':'Total'},axis=1)
  df[1].index = ['Cases']
  df[2] = df[2].rename({'Tot Deaths':'Total'},axis=1)
  df[2].index = ['Deaths']
  df = df[1].append(df[2]).T
  race = ['Total'] + ['Blk','White','AIAN','Asian','Other','Unk']*3
  eth = ['Total'] + ['NH']*6 + ['Hisp']*6 + ['Unk']*6
  df = df.set_index([eth,race])
  display(df)
  df_eth = df.sum(level=0).reset_index().rename({'index':'Category'},axis=1)
  df_race = df.loc[['NH','Unk']].sum(level=1).reset_index().rename({'index':'Category'},axis=1)
  df = pd.concat([df_race,df_eth]).reset_index(drop=True)
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runMT(path,date,state,keys):
  # Read state file(s)
  num_files = 3 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    display(df[i])
  # Pre-processing
  df_tot = df[2]
  i = 0
  for bool in df[1].loc[:,'Unnamed: 0'].str.isalpha():
    if bool:
      df[1].loc[i-1,'Cases']=df[1].loc[i,'Race']
      df[1].loc[i-1,'Deaths']=df[1].loc[i,'Cases']
    i+=1
  df=df[1][~df[1].loc[:,'Unnamed: 0'].str.isalpha()]
  df=df.drop(['Unnamed: 0'],axis=1).reset_index(drop=True)
  df.loc[len(df.index)] = ['Total',df_tot.loc[0,'Totals']+df_tot.loc[1,'Totals'],df_tot.loc[2,'Totals']]
  df.loc[len(df.index)] = ['Race Unknown',df.loc[11,'Cases']-df.loc[6,'Cases'],df.loc[11,'Deaths']-df.loc[6,'Deaths']]
  df.loc[len(df.index)] = ['Eth Unknown',df.loc[11,'Cases']-df.loc[10,'Cases'],df.loc[11,'Deaths']-df.loc[10,'Deaths']]
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runNC(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0','Percent of Cases','Percent of Deaths'],axis=1)
    df[i]=df[i].rename({'Unnamed: 0.1':'Category'},axis=1)
    df[i]=df[i].set_index('Category')
    df[i].loc['Total']=df[i].sum()
    display(df[i])
  # Pre-processing
#  df[i].loc[len(df[i].index)]=
  df = df[1].append(df[2]).reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runND(path,date,state,keys):
  # Read state file(s)
  num_files = 1 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df=df[1].set_index('Category')
  df.loc['Total']=df.sum()
  df=df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runNE(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df[1].index = ['Total']
  df[1].columns = ['Cases','Hospitalizations','Deaths']
  df[2].columns = ['Category','Cases','Hospitalizations','Deaths']
  df[2]=df[2].set_index('Category').drop('Category')
  for cat in ['Cases','Hospitalizations','Deaths']:
    df[2].loc[:,cat] = df[2].loc[:,cat].astype('float')
    df[2].loc[:,cat] = df[2].loc[:,cat].astype('int')
  df=df[1].append(df[2])
  df.loc['Known Eth']=df.iloc[1:3].sum()
  df.loc['Known Race']=df.iloc[4:11].sum()-df.loc['race_Unknown']
  df.loc['Unknown Eth']=df.loc['Total']-df.loc['Known Eth']
  df.loc['Unknown Race']=df.loc['Total']-df.loc['Known Race']
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runNH(path,date,state,keys):
  # Read state file(s)
  num_files = 5 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df_tot = df[4]
  df_tot.columns = ['Cases','Hospitalizations','Deaths']
  df_tot.index = ['Total']
  df_hosp=df[5].rename({'Race/Ethnicity':'Demographic','Hosp':'Hospitalizations'},axis=1)
  df_hosp=df_hosp.set_index('Demographic')
  df_hosp=df_hosp.rename({'Hispanic or Latino':'Hispanic/Latino','Total':'Known'})
  display(df_hosp)
  for i in range(1,3):
    df[i] = df[i].drop(['Sex', 'Suppression', 'Date'],axis=1)
  df_demo = df[1].merge(df[2],how='outer',on='Demographic')
  df_demo = df_demo.drop(0)
  df_demo = df_demo.set_index('Demographic')
  df_demo.index = df_demo.index.str.replace('^ ','', regex=True)
  df_demo.loc['Known']=df_demo.sum()
  df = df_demo.merge(df_hosp,how='outer',on='Demographic').fillna(0).astype('int')
  df = df.append(df_tot)
  df.loc['Unknown (calc)'] = df.loc['Total']-df.loc['Known']+df.loc['Unknown']
  df.loc['Other'] = df.loc['Other']+df.loc['Other Race']
  df.loc['NH'] = df.loc['Total']-df.loc['Hispanic/Latino']-df.loc['Unknown (calc)']
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runNM(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df[1].columns = ['Category','Cases']
  df[2].columns = ['Category','Deaths']
  for i in range(1,3):
    df[i]=df[i].set_index('Category')
  df[1].loc['unk (calc)']=df[1].loc['cases','Cases']-df[1].iloc[[4,5,6,7,9,10,11],0].sum()
  df[1].loc['NH']=df[1].loc['cases','Cases']-df[1].loc['hispanic','Cases']-df[1].loc['unk (calc)','Cases']
  df[2].loc['Total'] = df[1].loc['deaths','Cases']
  df[2].loc['unk (calc)']=df[2].loc['Total','Deaths']-df[2].iloc[0:5,0].sum()
  df[2].loc['NH']=df[2].loc['Total','Deaths']-df[2].loc['Hispanic','Deaths']-df[2].loc['unk (calc)','Deaths']
  df = df[1].merge(df[2],how='outer',on='Category').fillna(0).reset_index()
  for cat in ['Cases','Deaths']:
    df[cat] = df[cat].astype('int')
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runNV(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    df[i]=df[i].set_index('Category',drop=True)
    display(df[i])
  # Pre-processing
  df_tots=df[2].replace(',','', regex=True).astype('int')
  display(df_tots)
  df = df[1].replace('%','',regex=True).astype('float')
  display(df_tots.values*df.values/100)
  df = pd.DataFrame(df_tots.values*df.values/100,columns=df.columns,index=df.index).apply(np.floor).astype('int')
  df.loc['Known']=df.sum()
  df = df.append(df_tots)
  df.loc['Unknown']=df.loc['Totals']-df.loc['Known']
  df.loc['NH']=df.loc['Totals']-df.loc['Hispanic']-df.loc['Unknown']
  df=df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runOR(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    df[i]=df[i].rename(columns={'Categories (group) 2':'Category','Categories (group)':'Category'})
    df[i]=df[i].set_index(['Demographic','Category'])
    display(df[i])
  # Pre-processing
  hosp_tots=df[2].sum(level=0)
  df[2].loc[('Race','Total'),:]=hosp_tots.loc['Race']
  df[2].loc[('Ethnicity','Total'),:]=hosp_tots.loc['Ethnicity']
  df = df[1].merge(df[2],how='outer',on=['Demographic','Category'])
  df.columns=['Deaths','Cases','Hospitalizations']
  df['Hospitalizations']=df['Hospitalizations'].astype('int')
  df=df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runPA(path,date,state,keys):
  # Read state file(s)
  num_files = 5 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df[1] = df[1].set_index('Race').rename(columns={'Positive_Cases':'Cases'})
  df[2] = df[2].set_index('Ethnicity')
  df[3] = df[3].set_index('Race').rename(index={'African American':'Black'})
  df[4] = df[4].set_index('Ethnicity').rename(columns={'F__of_Deaths':'Deaths'})
  totals = [df[5].loc[0,'Cases'],df[5].loc[0,'Deaths']]
  display(totals)
  for i in range(1,5):
    if i%2 == 1:
      df[i].loc['Known Race']=df[i].sum()
    else:
      df[i].loc['Known Eth']=df[i].sum()
  df_cases = df[1].append(df[2]).reset_index().rename(columns={'index':'Category'})
  df_deaths = df[3].append(df[4]).reset_index().rename(columns={'index':'Category'})
  display(df_cases)
  display(df_deaths)
  df = df_cases.merge(df_deaths,how='left',on='Category').fillna(-1)
  df['Deaths']=df['Deaths'].astype('int')
  df.loc[len(df.index)]=['Total',totals[0],totals[1]]
  df.loc[len(df.index)]=['Unk Race',totals[0]-df.loc[5,'Cases']+df.loc[4,'Cases'],totals[1]-df.loc[5,'Deaths']]
  df.loc[len(df.index)]=['Unk Eth',totals[0]-df.loc[9,'Cases']+df.loc[6,'Cases'],totals[1]-df.loc[9,'Deaths']]
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runRI(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df_tots = df[1]
  df_tots = df_tots.loc[[df_tots.index[1]] + [df_tots.index[0]] + df_tots.index[2:].tolist()]
  display(df_tots)
  df=df[2].rename(columns=lambda c: 'Tests' if 'Tested' in c else c).set_index('Category')
  df.loc['Known']=df.sum()-df.loc['Totals']-df.loc['Unknown or pending further information']
  df.loc['Total']=df_tots.iloc[:,1].values
  df.loc['Unk (calc)']=df.loc['Total']-df.loc['Known']
  df.loc['NH']=df.loc['Known']-df.loc['Hispanic or Latino†']
  df=df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runSD(path,date,state,keys):
  # Read state file(s)
  num_files = 1 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df[1].columns = ['Category','Cases','Deaths','Hospitalizations']
  df = df[1].set_index('Category')
  df.loc['Total']=df.sum()
  df.loc['NH']=df.loc['Total']-df.loc['Hispanic']-df.loc['Unknown, Non-Hispanic']
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runTN(path,date,state,keys):
  # Read state file(s)
  num_files = 1 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df = df[1].set_index('Race').drop('Ethnicity').astype('int')
  df.loc['Total']=df.iloc[:7].sum()
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runTX(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0','%'],axis=1)
    df[i]=df[i].replace(' ','',regex=True)
    display(df[i])
  # Pre-processing
  df[1].columns = ['Category','Cases']
  df[2].columns = ['Category','Deaths']
  df = df[1].merge(df[2],how='outer',on='Category').set_index('Category')
  df.loc['NH'] = df.loc['Total']-df.loc['Hispanic']-df.loc['Unknown']
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runUT(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df[2]=df[2].rename(columns={'Persons Tested':'Tests'})
  df[2]=df[2].replace('Other Race','Other race')
  df= df[1].merge(df[2],how='outer',on='Category').set_index('Category')
  df.loc['NH'] = df.loc['Statewide']-df.loc['Hispanic or Latino']-df.loc['Unknown']
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runVA(path,date,state,keys):
  # Read state file(s)
  num_files = 3 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    if i == 3:
      df[i]=df[i].drop('Race/Ethnicity',axis=1)
    display(df[i])
  # Pre-processing
  df[3].columns = ['Category','Cases','Hospitalizations','Deaths']
  df[3]=df[3].set_index('Category')
  df = df[3]
  df.loc['Total']=df.sum()
  df.loc['NH'] = df.loc['Total']-df.loc['Latino']-df.loc['Not Reported']
  df = df.reset_index()
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def runVT(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  df = df[1].merge(df[2],how='outer',on='Category')
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

def template(path,date,state,keys):
  # Read state file(s)
  num_files = 2 ### Edit this to equal the number of files in the repo
  df = {}
  for i in range(1,num_files+1):
    df[i] = st_csv(i,path,date,state)
    df[i]=df[i].drop(['Unnamed: 0'],axis=1)
    display(df[i])
  # Pre-processing
  # Common processing
  df_st = state_common(df,keys,state)
  # Custom Mapping
  # Return ribbon
  return df_st

begin_run = time.time()

key = load_state_keys('crdt_key.csv')

#states_all = ["AK","AL","AR","CA","CO","CT","DC","DE","FL","GA","GU",
#          "HI","ID","IL","IN","KY","LA","MA","MD","ME","MI",
#          "MO","MN","MS","MT","NC","ND","NE","NH","NM","NV",
#          "NY","OR","PA","RI","SD","TN","TX","UT","VA","VT",
#          "WA","WI","WY"]

states_all = ["AK","AL","AR","CA","CO","CT","DC","DE","FL","GA","GU","HI","ID","IL","IN","KY","LA","MA","MD"]
#states = ["FL"]
date_str = datetime.datetime.now().strftime("%Y%m%d") 

args_list  = sys.argv[1:]

# Options
opts = "hays:"
long_opts = ["help", "all", "state", "yesterday"]
try:
    # Parsing argument
    args, vals = getopt.getopt(args_list, opts, long_opts)

    # checking each argument
    for arg, val in args:
        if arg in ("-h", "--help"):
            print ("Use:\n-a OR --all to run all states;\n-s <ST> OR --state <ST> to run a single state.")
            sys.exit()

        elif arg in ("-a", "--all"):
            print ("Running All")
            states = states_all

        elif arg in ("-s", "--state"):
            print (("Running State = (% s)") % (val))
            states = [str(val)]
            debug = True

        elif arg in ("-y", "--yesterday"):
            print ("Using Yesterday's Data")
            yesterday = datetime.datetime.now() - timedelta(1)
            date_str = datetime.datetime.strftime(yesterday,"%Y%m%d") 

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))


# You should not need to edit below this line

if len(states) == 0:
    sys.exit("ERROR: Must supply either -a or -s argumemnt")

# You should not need to edit below this line

gd_path = "/Users/User/Google Drive (patkellyatx@gmail.com)/CRDT"
first=True
failed = [-1]*52

for state in states:
  print("\n")
  display("***" + state + " Output:***")
  start = time.time()
  try:
    func = globals()["run" + state]
    df_st = func(gd_path+"/"+state,date_str,state,key)
    if first:
      df = df_st
      first = False
    else:
      df.loc[:,state]=df_st
    display("STATE PASSED")
  except Exception as e:
    display("STATE FAILED")
    display(e)
    if first:
      df = init_ribbon()
      df.loc[:,state]=failed
    else:
      df.loc[:,state]=failed
  end = time.time()
  duration = end - start
  print('%s run time = %.2f s' % (state, duration))


df_dates = pd.DataFrame([[date_str for i in range(len(df.columns))]],columns=df.columns,index=[['Date'],['']])
df = df_dates.append(df)
display(df)
df.T.to_csv('crdt_'+date_str+'.csv')


end_run = time.time()
time_run = end_run - begin_run
print('Total run time = %.2f s' % time_run )

