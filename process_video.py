import os
import subprocess
files=list(os.listdir("./videos_rag"))
for file in files:
    if "#" in file:
        tutorialnumber= file.split("#")[1].split(" ")[0]
        filename=file.split(" ｜ ")[0].replace(" ","")
        print(tutorialnumber, filename)
        os.makedirs("audios", exist_ok=True)

    subprocess.run(["ffmpeg","-i", f"videos_rag/{file}", f"audios/{tutorialnumber}_{filename}.mp3"])