# YouTube Video Processing and Analysis Application

## Overview

This Flask application provides a web interface for processing YouTube videos using various Language Models (LLMs) and embedding models. It supports multiple providers including Ollama, Groq, and Anthropic for text processing and analysis. Key features include:

- YouTube video transcript extraction
- Text processing with multiple LLM providers
- Vector embeddings for efficient text search
- Local storage of processed content
- Real-time status updates
- Dark mode UI
- Configuration management UI

## Requirements

- Python 3.8 or higher
- Flask
- youtube-transcript-api
- langchain
- pytube
- langchain_ollama
- requests
- yt_dlp
- faiss
- numpy
- anthropic
- asyncio
- groq

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/youtube-video-processing.git
   cd youtube-video-processing

2. pip install -r req.txt 

3.
export GROQ_API_KEY=your_groq_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export AIRTABLE_API_KEY=your_airtable_api_key

4. python app14.py



Endpoints 

    GET /config : Retrieve current configuration.
    POST /config/ollama_url : Update Ollama URL.
    GET /models : Retrieve available models from all providers.
    GET /models/groq : Retrieve available Groq models.
    GET /models/anthropic : Retrieve available Anthropic models.
    POST /process : Process a new YouTube video.
    GET /status : Check processing status of all videos.
    POST /query : Query processed video content using vector search.
     

Usage 

    Visit http://localhost:5001 in your browser.

1. Enter a YouTube URL 
2. select the desired LLM Provider
3. LLM
4. embedding model.

    Click "Process Video" to start processing.

    Check the status and results via the provided endpoints.
     
