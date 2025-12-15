import whisper
import os
import json
model = whisper.load_model("tiny")
audios=os.listdir("audios")

for audio in audios:
    number=audio.split("_")[0]
    title=audio.split("_")[1][:-4]
    print(number,title)
    result = model.transcribe(audio=f"audios\{audio}")
    print(result)

    chunks=[]
    for segment in result["segments"]:
        chunk={}
        chunk["number"]=number
        chunk["title"]=title
        chunk["start"]=segment["start"]
        chunk["end"]=segment["end"]
        chunk["text"]=segment["text"]
        chunks.append(chunk)
    chunkswithetxt={"chunks":chunks,"fulltext":result["text"]}
    os.makedirs("chunks", exist_ok=True)

    with open(f"chunks/{number}_{title}.json","w") as f:
        json.dump(chunkswithetxt,f)