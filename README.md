## RAG Project – Video Course QA Assistant

This project builds a simple Retrieval-Augmented Generation (RAG) pipeline on top of a video course.  
It lets you ask natural language questions (e.g. *“Where is CSS taught in this course?”*) and returns:

- **Which video(s)** contain the answer  
- **Approximate timestamps** where the topic is discussed  
- **A concise explanation** phrased like a helpful teaching assistant

Under the hood, it:

- Extracts audio from course videos
- Transcribes and chunks the speech
- Embeds chunks with an embedding model
- Uses similarity search to find the most relevant chunks
- Sends a context-aware prompt to a local LLM (via Ollama) to generate the final answer

---

## Project Structure

- **`videos_rag/`**: Original course videos (`.webm`)
- **`audios/`**: Extracted audio files (`.mp3`) generated from videos
- **`chunks/`**: JSON files with transcribed and chunked text, along with timestamps
- **`embedded_chunks.joblib`**: Precomputed embeddings and metadata for all chunks (Pandas DataFrame stored with `joblib`)
- **`vid_to_mp3.py`**: Converts videos in `videos_rag/` to `.mp3` files in `audios/`
- **`mp3_to_json.py`**: Uses Whisper to transcribe each audio file, producing chunked JSON in `chunks/`
- **`json_to_embeddings.py`**: Reads all JSON chunk files, generates embeddings via Ollama, and saves `embedded_chunks.joblib`
- **`process_incoming.py`**: Main query script – takes a user question, retrieves similar chunks, builds a prompt, calls the LLM, and writes the answer to `response.txt`
- **`prompt.txt`**: Example prompt content / debug artifact
- **`response.txt`**: Example model answer for a previous query

---

## Data and Naming Conventions

- Videos in `videos_rag/` are expected to follow the pattern used by the Sigma Web Dev Course, e.g.:  
  `Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].webm`
- `vid_to_mp3.py` turns each video into an audio file in `audios/` named like:  
  `2_YourFirstHTMLWebsite.mp3`
- `mp3_to_json.py` then creates corresponding JSONs in `chunks/` named like:  
  `2_YourFirstHTMLWebsite.json`

Each chunk in the JSON has:

- **`number`**: tutorial number (as a string)
- **`title`**: a compact title (e.g. `YourFirstHTMLWebsite`)
- **`start` / `end`**: timestamps (in seconds) within the video
- **`text`**: transcribed text segment

`json_to_embeddings.py` adds:

- **`id`**: a running chunk ID
- **`embedding`**: embedding vector for that chunk (from the `bge-m3` model)

All of this is stored in `embedded_chunks.joblib` as a DataFrame used by `process_incoming.py`.

---

## Requirements

### System Dependencies

- **Python 3.9+** (recommended)
- **ffmpeg** (needed by `vid_to_mp3.py` to extract audio)
- **Ollama** (running locally with models):
  - Embedding model: **`bge-m3`**
  - LLM model: **`llama3.2`** (or another compatible chat model)

Make sure `ffmpeg` is on your `PATH`, and Ollama is running and accessible at  
`http://localhost:11434/`.

### Python Packages

Install the required Python libraries (you can use a virtual environment):

```bash
pip install requests pandas numpy scikit-learn joblib openai-whisper
```

> **Note**: `openai-whisper` may require additional system packages (like `ffmpeg` and PyTorch) depending on your environment.

---

## End-to-End Pipeline

### 1. Convert Videos to MP3 – `vid_to_mp3.py`

Place your `.webm` (or compatible) videos in `videos_rag/` following the Sigma naming style.  
Then run:

```bash
python vid_to_mp3.py
```

This will:

- Iterate through all files in `videos_rag/`
- Derive a tutorial number and a compact title from the filename
- Call `ffmpeg` to create audio files in `audios/` as `<number>_<Title>.mp3`

### 2. Transcribe Audio and Create Chunks – `mp3_to_json.py`

This step uses Whisper to transcribe each `.mp3` and create JSON chunks with timestamps:

```bash
python mp3_to_json.py
```

What it does:

- Loads the Whisper **`small`** model
- For each audio in `audios/`:
  - Transcribes and (optionally) translates (`language="hi"`, `task="translate"` in the current code)
  - Builds a list of segments with `number`, `title`, `start`, `end`, and `text`
  - Saves them into `chunks/<number>_<Title>.json`

### 3. Generate Embeddings – `json_to_embeddings.py`

Before running this step, ensure:

- Ollama is running
- The **`bge-m3`** embedding model is available and pulled

Then run:

```bash
python json_to_embeddings.py
```

What it does:

- Lists all JSON files in `chunks/`
- For each file:
  - Loads `chunks` from JSON
  - Calls the Ollama `/api/embed` endpoint with `model="bge-m3"` on the text of each chunk
  - Attaches the resulting embedding vectors and a unique `id` to each chunk
- Aggregates everything into a Pandas DataFrame and saves it as `embedded_chunks.joblib`

### 4. Ask Questions – `process_incoming.py`

Finally, you can query the course content:

```bash
python process_incoming.py
```

This script:

- Loads `embedded_chunks.joblib` into a DataFrame
- Prompts you for a question via `input()`
- Uses `/api/embed` to embed your query with `bge-m3`
- Computes cosine similarity between the query and all chunk embeddings
- Takes the **top 30** most similar chunks
- Builds a prompt that:
  - Includes your question
  - Includes those top chunks as JSON with `number`, `title`, `text`, `start`, `end`
  - Instructs the model to:
    - Answer only based on these chunks
    - Mention **which video** and **approximate timestamps**
    - Sound like a friendly teaching assistant
    - Apologize and say it doesn’t know if no relevant info is found
- Sends this prompt to Ollama `/api/generate` with `model="llama3.2"` and saves the final answer to `response.txt`

---

## What Could Be Improved (Suggested Changes)

These are not yet implemented, but are recommended next steps:

- **Configuration & Paths**
  - Externalize model names, host URLs, and folder paths to a config file or environment variables.
  - Add checks to ensure folders like `videos_rag/`, `audios/`, `chunks/` and `embedded_chunks.joblib` exist before running.

- **`vid_to_mp3.py`**
  - Guard against files that don’t match the expected naming pattern (currently assumes `#` is present).
  - Consider skipping non-video files or logging them clearly instead of failing.

- **`mp3_to_json.py`**
  - Make language (`language="hi"`) and `task` (`"translate"`) configurable.
  - Add error handling around Whisper transcription (e.g. skip problematic files but continue others).

- **`json_to_embeddings.py`**
  - Add basic retry / error handling around the embedding API call.
  - Optionally process files in batches or show progress indicators for large datasets.

- **`process_incoming.py`**
  - Refactor into functions (e.g. `load_index()`, `embed_query()`, `retrieve_top_k()`, `build_prompt()`, `generate_answer()`).
  - Remove unreachable code (e.g. `print("Inference Response:", r.json())` after `return`).
  - Add CLI flags or a simple API instead of relying solely on `input()`.
  - Improve prompt wording / grammar slightly and make the template easy to edit (e.g. store in a separate file).

- **General**
  - Add a `requirements.txt` (or `pyproject.toml`) to pin dependencies.
  - Add simple logging instead of just `print` statements.
  - Optionally introduce tests for small pieces (e.g. filename parsing, similarity ranking).

---

## How to Use This Repo

1. **Install system dependencies** (`ffmpeg`, Ollama, Python).  
2. **Install Python packages** with `pip install ...` as shown above.  
3. **Place videos** in `videos_rag/`.  
4. Run, in order:
   - `python vid_to_mp3.py`
   - `python mp3_to_json.py`
   - `python json_to_embeddings.py`
   - `python process_incoming.py`
5. Type your question when prompted and then read the answer in `response.txt`.

You now have a minimal but functional RAG assistant specialized to your video course.
