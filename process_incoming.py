import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df=joblib.load("embedded_chunks.joblib")
def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed", json={
        "model":"bge-m3"
        ,"input":text_list
    })
    embedding=r.json()['embeddings']
    return embedding    


incoming_queries = input("Enter your query: ")
query_embedding = create_embedding([incoming_queries])[0]
# print("Query Embedding:", query_embedding)

print(np.vstack(df['embedding'].values))
print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding])
# print("Similarities:", similarities)
top_rusults=30
top_results_indices = np.argsort(similarities[:,0])[::-1][:top_rusults]
# print("Top 3 similar chunks indices:", top_results_indices)
new_df = df.iloc[top_results_indices]
print(new_df[['number','title', 'text']])
for index,item in new_df.iterrows():
    print(index,item["number"],item['title'],item['text'],item['start'],item['end'])