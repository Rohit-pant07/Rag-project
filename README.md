# RAG Project – Video Course QA Assistant

## Brief Summary

A **Retrieval-Augmented Generation (RAG) pipeline** that enables natural language Q&A over video course content. The system extracts audio from videos, transcribes speech, creates searchable embeddings, and uses similarity search with an LLM to answer questions about course content—including which videos contain answers and approximate timestamps.

---

## Detailed Overview

### What It Does

This project builds an intelligent Q&A assistant for video-based courses that:

- **Processes video content**: Converts course videos to audio, transcribes them, and creates searchable text chunks
- **Enables semantic search**: Uses embedding vectors to find relevant content based on meaning, not just keywords
- **Generates contextual answers**: Combines retrieved video chunks with an LLM to provide:
  - Which video(s) contain the answer
  - Exact timestamps (start–end) where topics are discussed
  - Concise explanations in a teaching assistant style

### Architecture & Pipeline

The system follows a **4-stage pipeline**:

1. **Video → Audio Extraction** (`vid_to_mp3.py`)
   - Extracts audio from `.webm` videos using `ffmpeg`
   - Parses tutorial numbers and titles from filenames
   - Outputs `.mp3` files to `audios/` directory

2. **Audio → Text Transcription** (`mp3_to_json.py`)
   - Uses OpenAI Whisper (`small` model) for speech-to-text
   - Translates Hindi audio to English (`language="hi"`, `task="translate"`)
   - Creates timestamped chunks with metadata (number, title, start, end, text)
   - Saves JSON files to `chunks/` directory

3. **Chunk Merging** (`merge_chunks.py`)
   - Merges 5 consecutive chunks into larger segments
   - Preserves timestamps (start from first chunk, end from last chunk)
   - Combines text content for better context
   - Outputs merged chunks to `merged_chunks/` directory

4. **Embedding Generation** (`json_to_embeddings.py`)
   - Generates embeddings for all merged chunks using Ollama's `bge-m3` model
   - Assigns unique IDs to each chunk
   - Stores embeddings and metadata in a Pandas DataFrame
   - Saves as `embedded_chunks.joblib` for fast retrieval

5. **Query Processing** (`process_incoming.py`)
   - Loads precomputed embeddings from `embedded_chunks.joblib`
   - Embeds user query using `bge-m3`
   - Computes cosine similarity to find top 30 most relevant chunks
   - Builds context-aware prompt with retrieved chunks
   - Generates answer using Ollama's `llama3.2` model
   - Saves response to `response.txt`

---

## Project Structure

```
rag_project/
├── videos_rag/              # Original course videos (.webm)
├── audios/                  # Extracted audio files (.mp3)
├── chunks/                  # Transcribed chunks (JSON files)
├── merged_chunks/           # Merged chunks (5 chunks per merged chunk)
├── embedded_chunks.joblib   # Precomputed embeddings (DataFrame)
│
├── vid_to_mp3.py           # Video → Audio extraction
├── mp3_to_json.py          # Audio → Text transcription
├── merge_chunks.py         # Chunk merging (5 chunks → 1)
├── json_to_embeddings.py   # Generate embeddings & save index
├── process_incoming.py     # Main Q&A query script
│
├── prompt.txt              # Example prompt (debug artifact)
└── response.txt            # Generated answer output
```

---

## Data Format

### Chunk Structure (chunks/)

Each JSON file contains:
```json
{
  "chunks": [
    {
      "number": "2",
      "title": "YourFirstHTMLWebsite",
      "start": 0.0,
      "end": 2.64,
      "text": "Transcribed text segment..."
    }
  ],
  "fulltext": "Complete transcription..."
}
```

### Merged Chunk Structure (merged_chunks/)

Merged chunks combine 5 consecutive chunks:
```json
{
  "chunks": [
    {
      "number": "2",
      "title": "YourFirstHTMLWebsite",
      "start": 0.0,
      "end": 10.8,
      "text": "Combined text from 5 chunks..."
    }
  ],
  "fulltext": "Complete transcription..."
}
```

### Embedded Chunks (embedded_chunks.joblib)

Pandas DataFrame with columns:
- `id`: Unique chunk identifier
- `number`: Tutorial number
- `title`: Video title
- `start`: Start timestamp (seconds)
- `end`: End timestamp (seconds)
- `text`: Chunk text content
- `embedding`: Embedding vector (from `bge-m3`)

---

## Requirements

### System Dependencies

- **Python 3.9+**
- **ffmpeg** (for video-to-audio conversion)
- **Ollama** (running locally at `http://localhost:11434/`)
  - Embedding model: `bge-m3`
  - LLM model: `llama3.2`

### Python Packages

```bash
pip install requests pandas numpy scikit-learn joblib openai-whisper
```

**Note**: `openai-whisper` requires additional system dependencies (ffmpeg, PyTorch) depending on your environment.

---

## Usage

### Step 1: Extract Audio from Videos

Place `.webm` videos in `videos_rag/` following the naming pattern:
```
Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].webm
```

Run:
```bash
python vid_to_mp3.py
```

**What it does**:
- Parses tutorial number from filename (e.g., `#2`)
- Extracts title (e.g., `Your First HTML Website`)
- Converts to `audios/2_YourFirstHTMLWebsite.mp3`

### Step 2: Transcribe Audio to Text

```bash
python mp3_to_json.py
```

**What it does**:
- Loads Whisper `small` model
- Transcribes each `.mp3` file (translates Hindi → English)
- Creates timestamped chunks
- Saves to `chunks/<number>_<Title>.json`

### Step 3: Merge Chunks

```bash
python merge_chunks.py
```

**What it does**:
- Reads all JSON files from `chunks/`
- Merges 5 consecutive chunks into larger segments
- Preserves start/end timestamps
- Saves merged chunks to `merged_chunks/`

### Step 4: Generate Embeddings

Ensure Ollama is running and `bge-m3` model is available:

```bash
python json_to_embeddings.py
```

**What it does**:
- Loads all merged chunk JSON files
- Generates embeddings via Ollama `/api/embed` endpoint
- Creates DataFrame with embeddings and metadata
- Saves to `embedded_chunks.joblib`

### Step 5: Query the Course

```bash
python process_incoming.py
```

**What it does**:
- Prompts for user question
- Embeds query using `bge-m3`
- Finds top 30 most similar chunks (cosine similarity)
- Builds prompt with:
  - User question
  - Top 30 chunks (number, title, text, start, end)
  - Instructions for LLM to answer based on chunks only
- Generates answer using `llama3.2`
- Saves response to `response.txt`

---

## Key Features

- **Semantic Search**: Uses embedding-based similarity search instead of keyword matching
- **Timestamped Answers**: Provides exact video timestamps where topics are discussed
- **Context-Aware**: Retrieves top 30 relevant chunks for comprehensive context
- **Teaching Assistant Style**: Generates friendly, educational responses
- **Local Processing**: Uses local Ollama models (no external API keys required)

---

## Technical Details

### Embedding Model
- **Model**: `bge-m3` (via Ollama)
- **Purpose**: Converts text chunks and queries into dense vector representations
- **Similarity Metric**: Cosine similarity

### LLM Model
- **Model**: `llama3.2` (via Ollama)
- **Purpose**: Generates natural language answers from retrieved chunks
- **Prompt Strategy**: Includes retrieved chunks as context with strict instructions to answer only from provided content

### Chunk Merging Strategy
- **Merge Size**: 5 consecutive chunks per merged chunk
- **Rationale**: Provides better context for embedding while maintaining reasonable granularity
- **Timestamp Preservation**: Start time from first chunk, end time from last chunk

### Retrieval Strategy
- **Top-K**: Retrieves top 30 most similar chunks
- **Similarity Threshold**: None (always returns top 30)
- **Ranking**: Cosine similarity between query embedding and chunk embeddings

---

## Limitations & Future Improvements

### Current Limitations

- **Hardcoded Parameters**: Model names, paths, and merge size are hardcoded
- **No Error Handling**: Limited error handling for API calls and file operations
- **Single Language**: Currently configured for Hindi-to-English translation only
- **No Progress Indicators**: No progress bars for long-running operations
- **Unreachable Code**: Some debug code remains in `process_incoming.py`

### Suggested Improvements

- **Configuration Management**
  - Externalize model names, URLs, and paths to config file or environment variables
  - Add validation for required directories and files

- **Error Handling**
  - Add retry logic for API calls
  - Handle missing files gracefully
  - Skip problematic files and continue processing

- **Flexibility**
  - Make language and translation task configurable
  - Support multiple video formats
  - Allow configurable merge size and top-K retrieval

- **Code Quality**
  - Refactor `process_incoming.py` into modular functions
  - Remove unreachable/debug code
  - Add logging instead of print statements

- **User Experience**
  - Add CLI flags for query input
  - Create simple API endpoint
  - Add progress indicators for long operations
  - Improve prompt template (store in separate file)

- **Testing**
  - Add unit tests for filename parsing
  - Test similarity ranking logic
  - Validate chunk merging correctness

---

## Example Workflow

1. **Setup**:
   ```bash
   # Install dependencies
   pip install requests pandas numpy scikit-learn joblib openai-whisper
   
   # Ensure Ollama is running with models
   ollama pull bge-m3
   ollama pull llama3.2
   ```

2. **Process Videos**:
   ```bash
   python vid_to_mp3.py          # Extract audio
   python mp3_to_json.py         # Transcribe
   python merge_chunks.py        # Merge chunks
   python json_to_embeddings.py  # Generate embeddings
   ```

3. **Query**:
   ```bash
   python process_incoming.py
   # Enter: "Where is CSS taught in this course?"
   # Check response.txt for answer
   ```

---

## Notes

- **Naming Convention**: Videos must follow the pattern with `#` for tutorial number extraction
- **Whisper Translation**: Currently configured to translate Hindi audio to English
- **Local Models**: Requires Ollama running locally; no cloud API dependencies
- **Storage**: `embedded_chunks.joblib` contains the complete searchable index

---

