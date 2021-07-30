import pandas as pd
import logging
logger = logging.getLogger(__name__)

##parallelization
import multiprocessing as mp

##journal list specific packages
import zipfile
#import re
from bs4 import BeautifulSoup
import os

def hello_world():
    print("i am a test")


def process_zipfile(zf_name):
    zf = zipfile.ZipFile(zf_name)
    names = zf.namelist()
    metanames = [s for s in names if 'metadata' in s]

    # bulid metadata
    metadata_list = []
    if_top6 = lambda journal: 1 if journal in top6_journals else 0
    for metaname in metanames:
        try:
            id_ = '-'.join([s.replace('.xml', '') for s in metaname.split('-')[2:]])
            soup = BeautifulSoup(zf.read(metaname), 'html.parser')
            journal = soup.find_all('journal-title')[0].text
            # abstract = soup.find_all('abstract')[0].text.lower().replace('\n','')
            # title = soup.find_all('article-title')[0].text.lower()
            year = soup.find('pub-date').findChild('year').text
            # top6 = if_top6(journal)
            data = {'jstor_id': id_,
                    'journal': journal,
                    'year': year,
                    # 'top6':top6,
                    # 'abstract':abstract,
                    # 'title':title
                    }
            metadata_list.append(data)
        except:
            pass

    meta_df = pd.DataFrame(metadata_list)
    print(zf_name)
    print('Done!')
    return meta_df


def multiprocessing_wrapper(input_data,
                            cores,
                            function):

    logger.info("CPU Cores used: " + str(cores))

    pool = mp.Pool(cores)
    results = pool.map(function, input_data)

    pool.close()
    pool.join()

    return results


def testfct(x):
    return pd.DataFrame(pd.DataFrame({"a": [x ** 2]}))


def journal_list_wrapper(target_addresses,
                         cores,
                         fct_process_zipfile = process_zipfile,
                         fct_multiprocessing_wrapper = multiprocessing_wrapper,
                         ):

    for count, address in enumerate(target_addresses):

        os.chdir(address)
        zfs = os.listdir(address)


        #pool = Pool(processes=1)
        #results = pool.map(process_zipfile, zfs)
        #pool.close()
        #pool.join()


        results = fct_multiprocessing_wrapper(input_data = zfs,
                                              cores = cores,
                                              function = fct_process_zipfile)

        #results = map(process_zipfile, zfs)
        logger.info(address)

        results_df = pd.concat(results).drop_duplicates(subset = ['jstor_id'])

        results_df['discipline'] = count

        dfs = []
        dfs.append(results_df)

    primary_df = pd.concat(dfs)

    counts_df = primary_df.groupby(by=["journal", "discipline"])["year"].value_counts().reset_index(name="count")

    return counts_df





def save_data_csv(dtf,
                  data_path,
                  output_file_name):
    new_file_name = data_path + '/' + output_file_name + '.csv'
    logging.info("SAVING NEW FILE: " + new_file_name)
    dtf.to_csv(new_file_name, index=False)









#############
# DEV
#############




def journal_list_wrapper_test(target_addresses,
                         cores,
                         fct_process_zipfile = testfct,
                         fct_multiprocessing_wrapper = multiprocessing_wrapper,
                         ):

    for count, address in enumerate(target_addresses):

        os.chdir(address)
        zfs = os.listdir(address)

        '''
        pool = Pool(processes=1)
        results = pool.map(process_zipfile, zfs)
        pool.close()
        pool.join()
        '''

        results = fct_multiprocessing_wrapper(input_data = zfs,
                                              cores = cores,
                                              function = fct_process_zipfile)

        #results = map(process_zipfile, zfs)
        logger.info(address)

        results_df = pd.concat(results).drop_duplicates(subset = ['jstor_id'])

        results_df['discipline'] = count

        dfs = []
        dfs.append(results_df)

    return dfs

    #primary_df = pd.concat(dfs)

    #counts_df = primary_df.groupby(by=["journal", "discipline"])["year"].value_counts()

    #return counts_df


def journal_list_wrapper_test2(target_addresses,
                              cores,
                              fct_process_zipfile=testfct,
                              fct_multiprocessing_wrapper=multiprocessing_wrapper,
                              ):

    results = fct_multiprocessing_wrapper(input_data=[1,2,3],
                                          cores=cores,
                                          function=testfct)

    return results




def process_zipfile_test(zf_name):

    return zf_name

    '''
    zf = zipfile.ZipFile(zf_name)
    return zf

    names = zf.namelist()
    metanames = [s for s in names if 'metadata' in s]

    # bulid metadata
    metadata_list = []
    if_top6 = lambda journal: 1 if journal in top6_journals else 0

    return metanames


    for metaname in metanames:
        try:
            id_ = '-'.join([s.replace('.xml', '') for s in metaname.split('-')[2:]])
            soup = BeautifulSoup(zf.read(metaname), 'html.parser')
            journal = soup.find_all('journal-title')[0].text
            # abstract = soup.find_all('abstract')[0].text.lower().replace('\n','')
            # title = soup.find_all('article-title')[0].text.lower()
            year = soup.find('pub-date').findChild('year').text
            # top6 = if_top6(journal)
            data = {'jstor_id': id_,
                    'journal': journal,
                    'year': year,
                    # 'top6':top6,
                    # 'abstract':abstract,
                    # 'title':title
                    }
            metadata_list.append(data)
        except:
            pass

    meta_df = pd.DataFrame(metadata_list)
    print(zf_name)
    print('Done!')
    return meta_df
    '''


def journal_list_wrapper_test3(target_addresses,
                         cores,
                         fct_process_zipfile = process_zipfile_test,
                         fct_multiprocessing_wrapper = multiprocessing_wrapper,
                         ):

    for count, address in enumerate(target_addresses):

        #os.chdir(address)
        print(os.getcwd())
        zfs = os.listdir(address).copy()

        '''
        pool = Pool(processes=1)
        results = pool.map(process_zipfile, zfs)
        pool.close()
        pool.join()
        '''

        results = fct_multiprocessing_wrapper(input_data = zfs,
                                              cores = cores,
                                              function = fct_process_zipfile)

        #results = map(process_zipfile, zfs)
        logger.info(address)

    return results

    #primary_df = pd.concat(dfs)

    #counts_df = primary_df.groupby(by=["journal", "discipline"])["year"].value_counts()

    #return counts_df