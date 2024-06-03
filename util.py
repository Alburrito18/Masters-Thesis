import pandas as pd
import json
from datasets import load_dataset,DatasetDict,load_from_disk, Dataset
import re
from langchain_community.document_loaders import WikipediaLoader
import numpy as np
#from easy_entrez import EntrezAPI
#from easy_entrez.parsing import xml_to_string
#import lxml.etree as etree
#from Bio import Entrez

#mock_data_to_txt()


def open_mock_data(mock_data_path = "data/mock_data_2.xlsx"):
    dfs = pd.read_excel(mock_data_path, sheet_name=None, index_col=None)

    raw_df = dfs["raw"]
    config_df = dfs["config"]
    summary_df = dfs["summarized"]

    return (raw_df,config_df,summary_df)

def mock_data_to_txt():
    raw_df,config_df,summary_df = open_mock_data(mock_data_path = '/workspace/data/llm-anamnesis/mock_data_2_big.xlsx')
    summary_df['patient_id'] = summary_df['patient_ids'].apply(lambda x: x[1:-1] if x[0]=='[' else x)
    summary_df['date'] = summary_df['dates'].apply(lambda x:x[1:-1])
    raw_df['patient_id'] = raw_df['patient_id'].astype(str)
    variables = list()
    for index,r in summary_df.iterrows():
        temp_df = raw_df.loc[(raw_df['patient_id']==r['patient_id'])&(raw_df['date']==r['date'])]
        text = ""
        for row in temp_df.iterrows():
            row = row[1]
            if row['metadata'] != np.nan:
                text += str(row['metadata'])
            text +=row['text']
        if text == "":
            continue
        variables.append((text,r['text']))
    return variables

def download_and_save_dataset(dataset_name="Gabriel/pubmed_swe"):
    dataset = load_dataset(dataset_name)
    train_testvalid = dataset['train'].train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    train_test_valid_dataset.save_to_disk("/workspace/data/"+dataset_name)

def wikipedia_scrape(search_word = 'Kärlkirurgi', n_docs = 300, titles = True, load = True):
    #'/workspace/data/Gabriel/pubmed_swe'
    reg = r'([\n]+=+|\s+=+)'
    if not titles:
        reg = r'\n+|(\=.*\=)'
    docs = WikipediaLoader(query=search_word, load_max_docs=n_docs,lang='sv').load()
    if load:
        dataset = load_from_disk('/workspace/data/wikipedia')
        for doc in docs:
            if dataset.filter(lambda x:x['title']==doc.metadata['title']).num_rows == 0:
                dataset = dataset.add_item({'title':doc.metadata['title'],'summary':doc.metadata['summary'],'text':re.sub(reg,' ',doc.page_content)})
    else:
        ds = dict()
        ds['title'] = []
        ds['summary'] = []
        ds['text'] = []
        #rint(dataset)#.column_names()#=['title','summary','text']
        for doc in docs:
            ds['title'].append(doc.metadata['title'])
            ds['summary'].append(doc.metadata['summary'])
            ds['text'].append(re.sub(reg,' ',doc.page_content))
        dataset = Dataset.from_dict(ds)
    dataset.save_to_disk('/workspace/data/wikipedia')
    
def pub_scrape():
    search_query = '("Swedish"[Language] AND "medline"[Filter]) AND (ffrft[Filter])'
    # getting search results for the query 
    '''
    entrez_api = EntrezAPI(
    'your-tool-name',
    'e@mail.com',
    # optional
    return_type='json'
)
    result = entrez_api.search(search_query, max_results=2,database='pubmed')
    records = entrez_api.fetch(result.data['esearchresult']['idlist'],2)
    print(result.data)
    print(xml_to_string(records.data)) 
    '''
    Entrez.email = 'albert.lund@vgregion.se'
    search_results = Entrez.read(Entrez.esearch(db="pubmed", term=search_query, retmax=2, usehistory="y"))
    handle = Entrez.efetch(db="pubmed", rettype="full", retmode="xml", retstart=0, retmax=2, webenv=search_results["WebEnv"], query_key=search_results["QueryKey"])
    print(handle.read())

if __name__=="__main__":
    pub_scrape()