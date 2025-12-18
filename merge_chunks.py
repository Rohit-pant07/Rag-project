import os 
import json
import math

n=5 #number of chunks to be merged

for filename in os.listdir("chunks"):
    if filename.endswith(".json"):
        filepath=os.path.join("chunks",filename)
        with open (filepath,"r",encoding="utf-8") as f:
            data=json.load(f)
            new_chunks=[]
            num_chunks=len(data["chunks"])
            num_groups=math.ceil(num_chunks/n)

            for i in range(num_groups):
                start_index=i*n
                end_index=min((i+1)*n, num_chunks)
                chunk_group=data["chunks"][start_index:end_index]
                new_chunks.append({
                    "number":data["chunks"][0]["number"],
                    "title":data["chunks"][0]["title"],
                    "start":chunk_group[0]["start"],
                    "end":chunk_group[-1]["end"],
                    "text":" ".join([chunk["text"] for chunk in chunk_group])
                })
                
            os.makedirs("merged_chunks", exist_ok=True)
            with open (f"merged_chunks/{filename}","w",encoding="utf-8") as json_file:
                json.dump({"chunks":new_chunks,"fulltext":data["fulltext"]},json_file, indent=4)