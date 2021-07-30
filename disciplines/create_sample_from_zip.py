#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import zipfile
import re
from bs4 import BeautifulSoup


# In[2]:

top6_journals = {#economics
    'The American Economic Review',
    'The Review of Economic Studies',
    'The Quarterly Journal of Economics',
    'Journal of Political Economy',
    'Econometrica',
    'The Economic Journal',
    #sociology
    'American Journal of Sociology',
    'The British Journal of Sociology',
    'Social Forces',
    'European Sociological Review',
    'American Sociological Review',
    'Social Problems',
    #political sciences
    'American Journal of Political Science',
    'The American Political Science Review',
    'British Journal of Political Science',
    'The Journal of Politics',
    'Political Analysis',
    'International Organization'}

# In[3]:


def process_zipfile(zf_name):
    zf = zipfile.ZipFile(zf_name)
    names = zf.namelist()
    metanames = [s for s in names if 'metadata' in s]

    #bulid metadata
    metadata_list = []
    if_top6 = lambda journal: 1 if journal in top6_journals else 0
    for metaname in metanames:
        try:
            id_ = '-'.join([s.replace('.xml','') for s in metaname.split('-')[2:]])
            soup = BeautifulSoup(zf.read(metaname), 'html.parser')
            journal = soup.find_all('journal-title')[0].text
            abstract = soup.find_all('abstract')[0].text.lower().replace('\n','')
            title = soup.find_all('article-title')[0].text.lower()
            year = soup.find('pub-date').findChild('year').text
            top6 = if_top6(journal)
            data = {'jstor_id':id_,
                    'journal':journal,
                    'year':year,
                    'top6':top6,
                    'abstract':abstract,
                    'title':title
                    }
            metadata_list.append(data)
        except:
            pass
   
    
    meta_df = pd.DataFrame(metadata_list)
    print(zf_name)
    print('Done!')
    return meta_df


# In[4]:


import os
import multiprocessing as mp


# In[5]:

'''
dfs = []
target_addresses = ['/cluster/work/lawecon/Data/jstor_econ/raw',
                    '/cluster/work/lawecon/Work/dcai/journal_articles/data/jstor/jstor_sociology',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/political_science',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/african_american_studies',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/american_indian_studies',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/criminology',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/law',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/public_policy',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/management',
                    '/cluster/work/lawecon/Work/JSTOR_metadata/history']
'''


dfs = []
target_addresses = ['C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/jstor_econ/raw',
                    'C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/jstor_sociology']



# In[6]:


def multiprocessing_wrapper(input_data,
                            function,
                            cores):

    #logger.info("CPU Cores used: " + str(cores))

    pool = mp.Pool(cores)
    results = pool.map(function, input_data)

    pool.close()
    pool.join()

    return results

def testfct(x):
    return pd.DataFrame(pd.DataFrame({"a": [x**2]}))

if __name__ == "__main__":
    for count, address in enumerate(target_addresses):
        os.chdir(address)
        zfs = os.listdir(address)
        print
        '''
        pool = Pool(processes=1)
        results = pool.map(process_zipfile, zfs)
        pool.close()
        pool.join()
        '''
        results = multiprocessing_wrapper(zfs, process_zipfile, 4)
        #results = map(process_zipfile, zfs)
        print(address)
        results_df = pd.concat(results).drop_duplicates(subset = ['jstor_id'])
        results_df['discipline'] = count
        dfs.append(results_df)


# In[7]:


    primary_df = pd.concat(dfs)


# In[8]:


    primary_df.to_csv('C:/Users/damdu/kDrive/Master/Masterarbeit/Ash/dev/data/econ_sociology_test.csv')


# In[ ]:




