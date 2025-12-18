import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed", json={
        "model":"bge-m3"
        ,"input":text_list
    })
    embedding=r.json()['embeddings']
    return embedding    

jsons =os.listdir("chunks")#list of all json files in chunks folder
mydicts=[]
chunk_id=0

for jsonfile in jsons:
    with open(f"chunks/{jsonfile}") as f:
        data=json.load(f)
    print(f"Processing file: {jsonfile}")
    embeddings=create_embedding([c['text'] for c in data['chunks']])    
    
    for i,chunk in enumerate(data["chunks"]):
        chunk_id+=1
        chunk["id"]=chunk_id
        chunk['embedding']=embeddings[i]
        
        mydicts.append(chunk)
df=pd.DataFrame.from_records(mydicts)
print(df.head())
joblib.dump(df,"embedded_chunks.joblib")
