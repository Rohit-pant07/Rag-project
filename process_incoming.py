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
def inference(prompt):
    r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
)
    return r.json()
 
    print("Inference Response:", r.json())

# you can use OpenAI's API for inference as shown below:

            # from openai import OpenAI
            # def inference_with_openai(prompt):
            #     client = OpenAI()
            #     response = client.responses.create(
            #     model="gpt-5",
            #     input=prompt
            #     )

            #     print(response.output_text)
            #     return response.output_text

            


incoming_queries = input("Enter your query: ")
query_embedding = create_embedding([incoming_queries])[0]
# print("Query Embedding:", query_embedding)

# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding])
# print("Similarities:", similarities)
top_rusults=30
top_results_indices = np.argsort(similarities[:,0])[::-1][:top_rusults]
# print("Top 3 similar chunks indices:", top_results_indices)
new_df = df.iloc[top_results_indices]
# print(new_df[['number','title', 'text']])
    # for index,item in new_df.iterrows():
    #     print(index,item["number"],item['title'],item['text'],item['start'],item['end'])
prompt=f"""
You are a helpful teaching assistant for a video-based course.

The user asked the following question:
"{incoming_queries}"

Below is information extracted from this course’s video content, including video title, timestamps, and spoken text:
{new_df[['number','title','text','start','end']].to_json(orient='records')}

Your task:
- Answer the user’s question using ONLY the information from these course videos.
- Identify the most relevant video title.
- Clearly mention the exact time range (start–end) where the topic is explained.
- Briefly summarize what is taught in that time range in simple language.
- Guide the user on which part of the video to watch.
- If the topic appears in multiple parts, mention the best one first.

Rules:
- Do NOT mention words like “chunks”, “JSON”, or “dataset”.
- Do NOT add external knowledge.
- Sound like a friendly teaching assistant.
- Keep the answer concise and focused.
- If the question cannot be answered using the provided video content, reply exactly:
  "Sorry, I don’t have that information in this course videos."

"""
# with open("prompt.txt","w") as f:
#     f.write(prompt)
response=inference(prompt)['response']
print("Response:", response)
with open("response.txt","w") as f:
    f.write(response)