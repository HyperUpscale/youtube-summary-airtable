"""
YouTube Video Processing and Analysis Application

This Flask application provides a web interface for processing YouTube videos using various Language Models (LLMs)
and embedding models. It supports multiple providers including Ollama, Groq, and Anthropic for text processing
and analysis.

Key Features:
- YouTube video transcript extraction
- Text processing with multiple LLM providers
- Vector embeddings for efficient text search
- Local storage of processed content
- Real-time status updates
- Dark mode UI
- Configuration management UI
"""

from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pytube import YouTube
from langchain_ollama import OllamaEmbeddings
import requests
import re
from datetime import datetime
import os
import json
from pathlib import Path
import yt_dlp
import logging
import faiss  # Import FAISS for GPU support
import numpy as np  # Import numpy for array operations
import requests
import anthropic
import asyncio
from groq import AsyncGroq

# Configuration management functions
def load_config():
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "ollama_urls": ["http://localhost:11434"],
        "current_ollama_url": "http://localhost:11434"
    }

def save_config(config):
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

# Load initial configuration
config = load_config()
OLLAMA_BASE_URL = config['current_ollama_url']
EMBEDDING_SIZE_THRESHOLD = "1000000000"  

# API keys for different LLM providers
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Airtable config
AIRTABLE_BASE_ID = "appGv6kCsg55CdnoB"
AIRTABLE_TABLE_NAME = "tbljSFwHBL8tbwOuK"
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

# Create necessary directories for storing templates and static files
if not os.path.exists('templates'):
    os.makedirs('templates')
if not os.path.exists('static'):
    os.makedirs('static')

# Store processed videos and their status
processed_videos = {}  # Stores video processing status and results
vector_store = None    # FAISS vector store for efficient text search

def sanitize_filename(filename):
    """
    Sanitizes a filename by replacing invalid characters with underscores.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename safe for filesystem operations
    """
    # Replace spaces with underscores and remove any invalid characters
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in filename)

def save_captions_and_article(video_id, video_title, channel_name, captions, article):
    """
    Saves video captions and processed article to local filesystem.
    
    Args:
        video_id (str): YouTube video ID
        video_title (str): Title of the video
        channel_name (str): Name of the YouTube channel
        captions (str): Raw video captions/transcript
        article (str): Processed article from LLM
    """
    base_dir = Path(__file__).resolve().parent
    captions_dir = base_dir / 'Captions' / channel_name
    articles_dir = base_dir / 'Articles' / channel_name
    
    captions_dir.mkdir(parents=True, exist_ok=True)
    articles_dir.mkdir(parents=True, exist_ok=True)
    
   
    sanitized_video_title = sanitize_filename(video_title)
    
    captions_file = captions_dir / f"{video_id} - {sanitized_video_title}.txt"
    article_file = articles_dir / f"{video_id} - {sanitized_video_title}.txt"
    
    try:
        with captions_file.open('w', encoding='utf-8') as f:
            f.write(captions)
    except Exception as e:
        print(f"Error writing captions file: {e}")
    
    try:
        with article_file.open('w', encoding='utf-8') as f:
            f.write(article)
    except Exception as e:
        print(f"Error writing article file: {e}")

def load_captions_and_article(video_id, channel_name, video_title):
    """
    Attempts to load previously processed captions and article from local storage.
    
    Args:
        video_id (str): YouTube video ID
        channel_name (str): Name of the YouTube channel
        video_title (str): Title of the video
        
    Returns:
        tuple: (captions, article) if found, (None, None) if not found
    """
    base_dir = Path(__file__).resolve().parent
    captions_dir = base_dir / 'Captions' / channel_name
    articles_dir = base_dir / 'Articles' / channel_name
    
    # Sanitize video title
    sanitized_video_title = sanitize_filename(video_title)
    
    captions_file = captions_dir / f"{video_id} - {sanitized_video_title}.txt"
    article_file = articles_dir / f"{video_id} - {sanitized_video_title}.txt"
    
    if captions_file.exists() and article_file.exists():
        with captions_file.open('r', encoding='utf-8') as f:
            captions = f.read()
        with article_file.open('r', encoding='utf-8') as f:
            article = f.read()
            
        # Update processed_videos with the local article
        if video_id not in processed_videos:
            processed_videos[video_id] = {}
        processed_videos[video_id].update({
            'status': 'LOCAL',
            'article': article
        })
            
        return captions, article
    return None, None

# Configuration endpoints
@app.route('/config', methods=['GET'])
def get_config():
    return jsonify(load_config())

@app.route('/config/ollama_url', methods=['POST'])
def update_ollama_url():
    data = request.json
    config = load_config()
    new_url = data.get('url')
    
    if new_url:
        if new_url not in config['ollama_urls']:
            config['ollama_urls'].append(new_url)
        config['current_ollama_url'] = new_url
        save_config(config)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

def get_groq_models():
    """
    Fetches available models from the Groq API.
    
    Returns:
        dict: Dictionary containing list of available LLM models from Groq
    """
    # Fetch models from Groq API
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        models = data.get('data', [])
        llm_models = [{'name': model['id']} for model in models]
        return {'llm_models': llm_models}
    return {'llm_models': []}

def get_anthropic_models():
    """
    Returns a list of available Anthropic Claude models.
    
    Returns:
        dict: Dictionary containing list of available Claude models
    """
    # Return list of available Claude models
    return {'llm_models': [
        {'name': 'claude-3-opus-latest'},
        {'name': 'claude-3-5-sonnet-latest'},
        {'name': 'claude-3-5-haiku-latest'}
    ]}

def get_ollama_models(base_url=OLLAMA_BASE_URL):
    """
    Fetches available models from local Ollama instance.
    
    Args:
        base_url (str): Base URL for Ollama API
        
    Returns:
        dict: Dictionary containing separate lists for LLM and embedding models
    """
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            # Separate models based on size
            llm_models = []
            embedding_models = []
            
            for model in models:
                model_info = {
                    'name': model.get('name', ''),
                    'size': model.get('size', 0)
                }
                
                # Check if it's an embedding model (less than 1GB)
                if model.get('size', 0) < EMBEDDING_SIZE_THRESHOLD:
                    embedding_models.append(model_info)
                else:
                    llm_models.append(model_info)
            
            return {
                'llm_models': llm_models,
                'embedding_models': embedding_models
            }
    except Exception as e:
        logging.error(f"Error fetching Ollama models: {str(e)}")
        return {'llm_models': [], 'embedding_models': []}

# HTML template with dark mode support and real-time status updates
template_path = os.path.join('templates', 'index.html')
with open(template_path, 'w') as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Video Processor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="/static/dark-mode.css">
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Configuration Section -->
        <div class="mb-8 p-4 bg-white rounded-lg shadow dark:bg-gray-800">
            <h2 class="text-xl font-bold mb-4 dark:text-white">Configuration</h2>
            <div class="mb-4">
                <label class="block text-gray-700 dark:text-gray-300 mb-2" for="ollamaUrl">
                    Ollama URL
                </label>
                <div class="flex space-x-2">
                    <select id="ollamaUrl" class="flex-grow p-2 border rounded dark:bg-gray-700 dark:text-white">
                    </select>
                    <input type="text" id="newOllamaUrl" placeholder="Add new URL" 
                           class="p-2 border rounded dark:bg-gray-700 dark:text-white">
                    <button onclick="addOllamaUrl()" 
                            class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                        Add
                    </button>
                </div>
            </div>
        </div>

        <!-- Dark Mode Toggle Button -->
        <button id="darkModeToggle" class="fixed top-4 right-4 bg-gray-800 text-white px-4 py-2 rounded-full shadow-lg">
            <svg id="darkModeIcon" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
            </svg>
        </button>

        <!-- Input Section -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-bold mb-4">Process New Video</h2>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700">YouTube URL</label>
                    <input type="text" id="videoUrl" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500">
                </div>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">LLM Provider</label>
                        <select id="llmProvider" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500">
                            <option value="ollama" selected>Ollama</option>
                            <option value="groq">Groq</option>
                            <option value="anthropic">Anthropic</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">LLM Model</label>
                        <select id="model" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500">
                            <option value="">Select a model</option>
                        </select>
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Embedding Model</label>
                    <select id="embeddingModel" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500">
                        <option value="nomic-embed-text">nomic-embed-text</option>
                    </select>
                </div>
                <button onclick="processVideo()" class="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                    Process Video
                </button>
            </div>
        </div>

        <!-- Status Section -->
        <div id="status" class="bg-white rounded-lg shadow-md p-6 mb-6"></div>

        <!-- Article Section -->
        <div id="article" class="bg-white rounded-lg shadow-md p-6 mb-6"></div>

        <!-- Side Menu -->
        <div id="sideMenu" class="fixed top-0 right-0 h-full w-64 bg-white shadow-lg p-4 transform translate-x-full transition-transform duration-300 ease-in-out">
            <button id="closeSideMenu" class="absolute top-2 right-2 text-gray-500 hover:text-gray-700">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
            <h2 class="text-xl font-bold mb-4">Query FAISS Index</h2>
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700">Number of Files: <span id="numFiles">0</span></label>
                <label class="block text-sm font-medium text-gray-700">Number of Indexes: <span id="numIndexes">0</span></label>
            </div>
            <div class="mb-4">
                <input type="text" id="queryInput" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500" placeholder="Enter query">
                <button onclick="queryFaiss()" class="mt-2 bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600">
                    Query
                </button>
            </div>
            <div id="queryResults" class="border-t border-gray-200 pt-4"></div>
        </div>

        <!-- Toggle Side Menu Button -->
        <button id="toggleSideMenu" class="fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-full shadow-lg">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
        </button>
    </div>

    <script>
        
   


            
            // Update model options when provider changes
        document.getElementById('llmProvider').addEventListener('change', updateModelOptions);
        
        // Initial load of models
        updateModelOptions();

        async function updateModelOptions() {
            const provider = document.getElementById('llmProvider').value;
            const modelSelect = document.getElementById('model');
            const embeddingSelect = document.getElementById('embeddingModel');
            
            modelSelect.innerHTML = '';
            
            if (provider === 'ollama') {
                try {
                    const response = await fetch('/api/tags');
                    const data = await response.json();
                    
                    // Add default option for LLM models
                    modelSelect.innerHTML = '<option value="">Select a model</option>';
                    
                    // Add LLM models
                    if (data.llm_models) {
                        data.llm_models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.name;
                            option.textContent = `${model.name} (${(model.size / 1e9).toFixed(1)}GB)`;
                            modelSelect.appendChild(option);
                        });
                    }
                    
                    // Update embedding model options
                    embeddingSelect.innerHTML = '<option value="nomic-embed-text">nomic-embed-text</option>';
                    if (data.embedding_models) {
                        data.embedding_models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.name;
                            option.textContent = `${model.name} (${(model.size / 1e6).toFixed(1)}MB)`;
                            embeddingSelect.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Error fetching Ollama models:', error);
                    modelSelect.innerHTML = '<option value="">Error loading models</option>';
                }
            } else if (provider === 'groq') {
                try {
                    const response = await fetch('/models/groq');
                    const data = await response.json();
                    
                    // Add default option for LLM models
                    modelSelect.innerHTML = '<option value="">Select a model</option>';
                    
                    // Add LLM models
                    if (data.llm_models) {
                        data.llm_models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.name;
                            option.textContent = `${model.name}`;
                            modelSelect.appendChild(option);
                        });
                    }
                    
                    // Update embedding model options
                    embeddingSelect.innerHTML = '<option value="nomic-embed-text">nomic-embed-text</option>';
                } catch (error) {
                    console.error('Error fetching Groq models:', error);
                    modelSelect.innerHTML = '<option value="">Error loading models</option>';
                }
            } else if (provider === 'anthropic') {
                try {
                    const response = await fetch('/models/anthropic');
                    const data = await response.json();
                    
                    // Add default option for LLM models
                    modelSelect.innerHTML = '<option value="">Select a model</option>';
                    
                    // Add LLM models
                    if (data.llm_models) {
                        data.llm_models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.name;
                            option.textContent = `${model.name}`;
                            modelSelect.appendChild(option);
                        });
                    }
                    
                    // Update embedding model options
                    embeddingSelect.innerHTML = '<option value="nomic-embed-text">nomic-embed-text</option>';
                } catch (error) {
                    console.error('Error fetching Anthropic models:', error);
                    modelSelect.innerHTML = '<option value="">Error loading models</option>';
                }
            }
        }

        async function processVideo() {
            const url = document.getElementById('videoUrl').value;
            const provider = document.getElementById('llmProvider').value;
            const model = document.getElementById('model').value;
            const embeddingModel = document.getElementById('embeddingModel').value;

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        url, 
                        llm_provider: provider, 
                        model,
                        embedding_model: embeddingModel 
                    }),
                });
                const data = await response.json();
                if (data.status === 'success') {
                    updateStatus();
                } else {
                    alert('Error processing video: ' + data.message);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                const statusDiv = document.getElementById('status');
                const articleDiv = document.getElementById('article');
                
                if (data.videos && data.videos.length > 0) {
                    const videoInfo = data.videos[0];  // Get the most recent video
                    statusDiv.innerHTML = `
                        <h3 class="text-lg font-semibold mb-2">Status</h3>
                        <p><strong>Channel:</strong> ${videoInfo.channel_name}</p>
                        <p><strong>Title:</strong> ${videoInfo.video_title}</p>
                        <p><strong>Video ID:</strong> ${videoInfo.video_id}</p>
                        <p><strong>Status:</strong> ${videoInfo.status}</p>
                        ${videoInfo.error ? `<p class="text-red-500">Error: ${videoInfo.error}</p>` : ''}
                    `;
                    
                    if (videoInfo.article) {
                        articleDiv.innerHTML = `
                            <h3 class="text-lg font-semibold mb-2">Article</h3>
                            <div class="whitespace-pre-wrap">${videoInfo.article}</div>
                        `;
                    }
                }
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }
        
        // Toggle Side Menu
        document.getElementById('toggleSideMenu').addEventListener('click', () => {
            const sideMenu = document.getElementById('sideMenu');
            sideMenu.classList.toggle('translate-x-full');
        });

        // Close Side Menu
        document.getElementById('closeSideMenu').addEventListener('click', () => {
            const sideMenu = document.getElementById('sideMenu');
            sideMenu.classList.add('translate-x-full');
        });

        // Query FAISS Index
        async function queryFaiss() {
            const queryInput = document.getElementById('queryInput').value;
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: queryInput }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                const queryResults = document.getElementById('queryResults');
                queryResults.innerHTML = '';
                data.results.forEach(result => {
                    const resultDiv = document.createElement('div');
                    resultDiv.innerHTML = `<p>${result}</p>`;
                    queryResults.appendChild(resultDiv);
                });
            } else {
                alert('Error querying FAISS index: ' + data.message);
            }
        }

        // Dark Mode Toggle
        document.getElementById('darkModeToggle').addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const darkModeIcon = document.getElementById('darkModeIcon');
            if (document.body.classList.contains('dark-mode')) {
                darkModeIcon.innerHTML = `
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>
                `;
            } else {
                darkModeIcon.innerHTML = `
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
                `;
            }
        });
        
        // Load configuration on page load
        async function loadConfig() {
            const response = await fetch('/config');
            const config = await response.json();
            const urlSelect = document.getElementById('ollamaUrl');
            
            // Clear existing options
            urlSelect.innerHTML = '';
            
            // Add all URLs
            config.ollama_urls.forEach(url => {
                const option = document.createElement('option');
                option.value = url;
                option.text = url;
                if (url === config.current_ollama_url) {
                    option.selected = true;
                }
                urlSelect.appendChild(option);
            });
        }

        // Add new Ollama URL
        async function addOllamaUrl() {
            const newUrl = document.getElementById('newOllamaUrl').value.trim();
            if (!newUrl) return;
            
            const response = await fetch('/config/ollama_url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: newUrl })
            });
            
            if (response.ok) {
                document.getElementById('newOllamaUrl').value = '';
                await loadConfig();
            }
        }

        // Update current Ollama URL when selection changes
        document.getElementById('ollamaUrl').addEventListener('change', async (e) => {
            const response = await fetch('/config/ollama_url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: e.target.value })
            });
            
            if (response.ok) {
                await loadConfig();
            }
        });

        // Load config when page loads
        document.addEventListener('DOMContentLoaded', loadConfig);

        // Initial status update
        updateStatus();
        
        // Update status every 5 seconds
        setInterval(updateStatus, 5000);
    </script>
</body>
</html>
""")

async def convert_with_groq(text, model):
    """
    Processes text using Groq's LLM API.
    
    Args:
        text (str): Input text to process
        model (str): Name of the Groq model to use
        
    Returns:
        str: Processed text from the model
    """
    client = AsyncGroq(api_key=GROQ_API_KEY)
    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": f"Convert into a well-structured notes and (remove any advertising). Keep and stick to the text as much as possible. This is the text: {text}. Make the main points like essential knowledge and well-organized with bold format the key points and headers and paragraphs. Also use bullet points. Use HTML formatting."}
        ],
        model=model,
        stream=True
    )

    full_response = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
    return full_response

def convert_with_anthropic(text, model):
    """
    Processes text using Anthropic's Claude models.
    
    Args:
        text (str): Input text to process
        model (str): Name of the Claude model to use
        
    Returns:
        str: Processed text from the model
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = "You are an expert at converting video transcripts into well-structured notes. Keep the main points and essential knowledge while maintaining accuracy."
    
    try:
        message = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"Convert this video transcript into well-structured notes. Remove any advertising but keep all important information: {text}"
            }]
        )
        # Extract only the text content from the response
        content = message.content
        if hasattr(content, 'text'):
            return content.text
        elif isinstance(content, list) and content and hasattr(content[0], 'text'):
            return content[0].text
        else:
            return str(content)
    except Exception as e:
        logging.error(f"Error in Anthropic conversion: {str(e)}")
        raise

@app.route('/models')
def get_models_endpoint():
    """
    API endpoint to get available models from all providers.
    
    Returns:
        json: Dictionary containing available models from all providers
    """
    ollama_models = get_ollama_models()
    groq_models = get_groq_models()
    anthropic_models = get_anthropic_models()

    models = {
        'ollama': ollama_models,
        'groq': groq_models,
        'anthropic': anthropic_models
    }

    logging.info(f"Returning models: {models}")
    return jsonify(models)

@app.route('/models/groq')
def get_groq_models_endpoint():
    """
    API endpoint to get available Groq models.
    
    Returns:
        json: Dictionary containing available Groq models
    """
    models = get_groq_models()
    logging.info(f"Returning Groq models: {models}")
    return jsonify(models)

@app.route('/models/anthropic')
def get_anthropic_models_endpoint():
    """
    API endpoint to get available Anthropic models.
    
    Returns:
        json: Dictionary containing available Anthropic models
    """
    models = get_anthropic_models()
    logging.info(f"Returning Anthropic models: {models}")
    return jsonify(models)

def get_anthropic_models():
    """
    Returns a list of available Anthropic Claude models.
    
    Returns:
        dict: Dictionary containing list of available Claude models
    """
    # Return list of available Claude models
    return {'llm_models': [
        {'name': 'claude-3-opus-latest'},
        {'name': 'claude-3-5-sonnet-latest'},
        {'name': 'claude-3-5-haiku-latest'}
    ]}

@app.route('/api/tags')
def get_ollama_tags():
    """
    API endpoint to get available Ollama models.
    
    Returns:
        json: Dictionary containing available Ollama models
    """
    return jsonify(get_ollama_models())

def save_to_airtable(channel_name, video_title, article):
    """
    Saves processed article to Airtable.
    
    Args:
        channel_name (str): YouTube channel name
        video_title (str): Video title
        article (str): Processed article content
    """
    if not AIRTABLE_API_KEY:
        logging.error("Airtable API key not configured")
        return

    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "records": [{
            "fields": {
                "Channel": channel_name,
                "VideoTitle": video_title,
                "Summary": article
            }
        }]
    }

    try:
        response = requests.post(AIRTABLE_URL, headers=headers, json=data)
        response.raise_for_status()
        logging.info(f"Successfully saved to Airtable: {response.status_code}")
    except Exception as e:
        logging.error(f"Error saving to Airtable: {str(e)}")

@app.route('/process', methods=['POST'])
def process_video():
    """
    Main endpoint for processing YouTube videos.
    Handles video download, transcript extraction, and text processing.
    
    Returns:
        json: Processing status and video ID
    """
    data = request.json
    video_url = data['url']
    llm_provider = data['llm_provider']
    model = data['model']
    embedding_model = data['embedding_model']
    
    try:
        # Extract video ID
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', video_url)
        if not video_id_match:
            return jsonify({'status': 'error', 'message': 'Invalid YouTube URL'}), 400
        
        video_id = video_id_match.group(1)
        
        # Extract channel name and video title using yt-dlp
        ydl_opts = {}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            channel_name = info_dict.get('uploader', 'Unknown Channel')
            video_title = info_dict.get('title', 'Unknown Title')
        
        # Initialize status
        processed_videos[video_id] = {
            'url': video_url,
            'status': 'Processing',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'llm_provider': llm_provider,
            'model': model,
            'embedding_model': embedding_model,
            'channel_name': channel_name,
            'video_title': video_title
        }
        
        # Get subtitles
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        
        # Process and index in a non-blocking way
        from threading import Thread
        thread = Thread(target=process_and_index, args=(video_id, text, llm_provider, model, embedding_model, channel_name, video_title))
        thread.start()
        
        return jsonify({'status': 'success', 'video_id': video_id}), 200
        
    except Exception as e:
        if 'video_id' in locals():
            processed_videos[video_id]['status'] = 'Failed'
            processed_videos[video_id]['error'] = str(e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status')
def get_status():
    """
    Endpoint to check processing status of all videos.
    
    Returns:
        json: Current processing status and results if available
    """
    # Convert all stored articles to strings to ensure JSON serialization
    videos_status = []
    for info in processed_videos.values():
        status_entry = {
            'channel_name': str(info['channel_name']),
            'video_title': str(info['video_title']),
            'video_id': str(info['video_id']),
            'status': str(info['status'])
        }
        if info['status'] == 'Completed' or info['status'] == 'Completed (Local)':
            status_entry['article'] = str(info.get('article', '')).replace('[TextBlock(text=\'', '').replace('\', type=\'text\')]', '')
        else:
            status_entry['article'] = ''
        videos_status.append(status_entry)
    
    return jsonify({
        'videos': videos_status
    })

@app.route('/query', methods=['POST'])
def query():
    """
    Endpoint for querying processed video content using vector search.
    
    Returns:
        json: Search results with relevant text segments
    """
    data = request.json
    query_text = data['query']
    
    if vector_store is None:
        return jsonify({'status': 'error', 'message': 'No indexed content available'})
    
    results = vector_store.similarity_search(query_text, k=3)
    return jsonify({'status': 'success', 'results': [doc.page_content for doc in results]})

def process_and_index(video_id, text, llm_provider, model, embedding_model, channel_name, video_title):
    """
    Core function for processing video text and creating searchable index.
    
    Args:
        video_id (str): YouTube video ID
        text (str): Raw video transcript
        llm_provider (str): Selected LLM provider
        model (str): Selected LLM model
        embedding_model (str): Selected embedding model
        channel_name (str): YouTube channel name
        video_title (str): Video title
    """
    try:
        # Sanitize video title for display
        sanitized_title = re.sub(r'[^\w\s-]', '', video_title)
        
        # Initialize status
        processed_videos[video_id] = {
            'channel_name': channel_name,
            'video_title': sanitized_title,
            'video_id': video_id,
            'status': 'Checking local files'
        }

        # Check if the video has already been processed
        captions, article = load_captions_and_article(video_id, channel_name, video_title)
        if captions and article:
            processed_videos[video_id].update({
                'status': 'Completed (Local)',
                'article': article
            })
            logging.info(f"Found local files for video {video_id}, loading cached version")
            return article

        # If not in local files, proceed with processing
        processed_videos[video_id]['status'] = 'Processing'

        # Convert text based on selected provider
        if llm_provider == 'ollama':
            article = convert_with_ollama(text, model)
        elif llm_provider == 'groq':
            article = asyncio.run(convert_with_groq(text, model))
        elif llm_provider == 'anthropic':
            article = convert_with_anthropic(text, model)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Update status with completion and store article
        processed_videos[video_id].update({
            'status': 'Completed',
            'article': article
        })

        # Save the processed content
        save_captions_and_article(video_id, video_title, channel_name, text, article)
        # Add this right after save_captions_and_article() call
        save_to_airtable(channel_name, video_title, article)

        return article

    except Exception as e:
        logging.error(f"Error processing video {video_id}: {str(e)}")
        if video_id in processed_videos:
            processed_videos[video_id]['status'] = f'Error: {str(e)}'
        raise

def convert_with_ollama(text, model):
    """
    Processes text using local Ollama instance.
    
    Args:
        text (str): Input text to process
        model (str): Name of the Ollama model to use
        
    Returns:
        str: Processed text from the model
    """
    full_response = ""
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': model,
            'prompt': f'Convert into a well-structured notes and (remove any advertising). Keep and stick to the text as much as possible. This is the text: """\n{text}\n\n """. FOR THAT TEXT:  Make the main points like essential knowledge and well-organized with bold format the key points and headers and paragraphs. Also use bullet points. Use HTML formatting.',
            'options': {
                'num_ctx': 8192}
            
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line)
            if 'response' in json_response:
                full_response += json_response['response']
    
    return full_response
    
@app.route('/')
def home():
    """Renders the main application page"""
    return render_template('index.html')

if __name__ == '__main__':
    # Verify template exists before starting server
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found at {template_path}")
    # Start Flask development server
    app.run(debug=True, host='0.0.0.0', port=5001)
