from ankipandas import Collection
import re
import pandas as pd
import os
import zipfile
import io
reg = r"(<[^>]*>)|(&nbsp;)|(—&gt;)|(&amp;)"
pre_path = '../../../data/finetuning/hus75/unzipped'
directory = os.fsencode(pre_path)

data = list()
err = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    #anki_col = zipfile.Path(pre_path+'/'+filename, at='collection.anki2')
    if filename.endswith(".apkg"):
        col = Collection(pre_path+'/'+filename+'/collection.anki2')
        #data.append((file,file))
        for note in col.notes.iterrows():
            question = note[1]['nflds'][0]
            try:
                answer = note[1]['nflds'][1]
            except:
                err += 1
                continue
                #print(note[1]['nflds'])
                #print(filename)
                #raise
            if re.match(r"(<img src)",question) or re.match(r"(<img src)",answer):
                continue
            question = re.sub(reg,' ',question)
            answer = re.sub(reg,' ',answer)
            if re.match(r"^\s*$",question) or re.match(r"^\s*$",answer):
                continue
            data.append((question,answer))
        print(file)
print(err)
df = pd.DataFrame(data,columns = ['Question','Answer'])
df.to_csv('test.csv')
#df.to_json('test.json',compression=None)