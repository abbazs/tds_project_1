Build a self-contained FastAPI application called "TDS Virtual TA" that acts as a local AI assistant for the "Tools in Data Science" (TDS) course at IIT Madras.

General Requirements:
Use FastAPI for the API framework.

Use https://aipipe.org/ for generating answers.

Learn how to use the api from https://github.com/sanand0/aipipe?tab=readme-ov-file#api

Include a rule-based fallback using a keyword-matching knowledge base.

Implement a response caching system using an in-memory LRU cache.

Expose the following endpoints:

/: root info and version

/api/: POST endpoint that accepts student questions and returns smart responses

/health: health check endpoint with diagnostics and model loading info

Smart Response Logic:
First check cache.

If not cached, match question using a rule-based knowledge base.

If no rule matched, generate a response using the local Qwen model.

Fallback to a default rule-based message if model fails.

Use the Following Knowledge Base Topics:
"ai_model": GPT model policy for TDS assignments

"fastapi": how to deploy FastAPI to Railway

"pydantic": validation and common issues

API Models:
Define request and response models using Pydantic:

QuestionRequest: accepts a question string and optional file attachments.

AnswerResponse: contains answer string, list of links (url + text), and metadata like source and model name.

HealthResponse: for /health, include status, version, capabilities, and warnings/issues.

Config:
Use pydantic-settings to configure:

Model name

Generation parameters (temperature, top_p, repetition_penalty, max_new_tokens)

Enable/disable cache

Host and port settings

Caching:
Build a custom ResponseCache class using an in-memory dictionary and LRU-style deque tracking.

Use MD5 hashes of the question string as keys.

Dockerfile:
Base image: python:3.13-slim

Copy in pyproject.toml, uv.lock, and app/ folder

Use uv to install dependencies in a virtual environment.

Expose port 80

Run uvicorn using .venv/bin/uvicorn and bind to 0.0.0.0:$PORT

Structure:
Project should have this structure:

bash
Copy
Edit
📁 tds_project_1/
└── 📁 app/
    ├── Dockerfile
    ├── __init__.py        # Main FastAPI app
    ├── config.py          # Pydantic settings
    ├── models.py          # API models
    └── utils.py           # Cache, rule-based engine, health checker
Include realistic examples in schema metadata and logging messages with ✅, ❌, 📚, etc.
Do not use external databases or authentication.
Ensure it runs completely standalone using just uv, FastAPI, and Pydantic.