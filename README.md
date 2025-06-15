# TDS Virtual TA

A Virtual Teaching Assistant for the IIT Madras BS in Data Science program's "Tools in Data Science" course, designed to answer student questions based on course content and Discourse posts.

## Overview

This project fulfills the requirements of Project 1 for the IIT Madras BS in Data Science program's "Tools in Data Science" course (TDS Jan 2025). It provides an API that automatically responds to student questions by leveraging:

- Course content from [TDS Jan 2025](https://tds.s-anand.net/#/2025-01/) as of April 15, 2025.
- Discourse posts from the [TDS Knowledge Base](https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34) between January 1, 2025, and April 14, 2025.

The application scrapes data, generates embeddings for semantic search, and uses OpenAI's models to provide concise, relevant answers with source links. It supports text questions with optional base64-encoded image attachments.

## Features

- **Data Scraping**:
  - Scrapes markdown course content from the `data/course` directory.
  - Scrapes Discourse posts within a specified date range using authenticated API requests (requires cookies).
  - Extracts and analyzes images in both course markdown files and Discourse posts using OpenAI's `gpt-4o-mini` vision model.

- **Embedding Generation**:
  - Processes markdown and JSON data into chunks using a custom `URLAwareTextSplitter`.
  - Generates embeddings with OpenAI's `text-embedding-3-small` model.
  - Merges embeddings into a single `embeddings.npz` file for efficient retrieval.

- **API Endpoint**:
  - Exposes a FastAPI endpoint at `/api/` for POST requests with JSON payloads containing a question and optional base64 image.
  - Returns a JSON response with a concise answer (under 150 words) and up to three relevant source links within 30 seconds.
  - Uses semantic search with precomputed embeddings to find relevant passages, enhanced by image context if provided.

- **Rate Limiting**:
  - Implements robust rate limiting for OpenAI API calls (500 RPM for vision, 3000 RPM for embeddings) to ensure reliability under load.

- **CLI Tools**:
  - Provides a command-line interface for scraping, embedding, and self-updating the `pyproject.toml` scripts.
  - Commands include `scrape`, `embed`, and `self-update` for managing data and embeddings.

## Project Structure

```
app/
├── __init__.py               # CLI entry point and self-update logic
├── api/
│   └── __init__.py           # FastAPI endpoint for question answering
├── concise_answer.py         # OpenAI-based concise answer generation
├── data/
│   ├── __init__.py           # Data scraping CLI
│   ├── course.py             # Course markdown scraping and image analysis
│   └── discourse.py          # Discourse post scraping and image analysis
├── embed/
│   ├── __init__.py           # Embedding CLI
│   ├── course.py             # Course content embedding
│   ├── discourse.py          # Discourse post embedding
│   ├── split.py              # URL-aware text splitting
│   └── utils.py              # Embedding utilities
├── embedder.py               # OpenAI embedding generation
├── image_context.py          # OpenAI image analysis
├── models.py                 # Pydantic models for settings and responses
└── utils/
    └── __init__.py           # Shared utilities (logging, table printing)
```

## Installation

1. **Install Astral UV**:
   Ensure `uv` is installed. Follow instructions at [Astral UV](https://github.com/astral-sh/uv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/tds-virtual-ta.git
   cd tds-virtual-ta
   ```

3. **Set Up Virtual Environment and Install Dependencies**:
   Create a virtual environment and install dependencies using `uv sync`:
   ```bash
   uv sync
   ```

4. **Set Environment Variables**:
   Create a `.env` file in the project root:
   ```bash
   echo "API_KEY=your-openai-api-key" > .env
   ```

5. **Configure Discourse Scraping**:
   Create a `config.json` for Discourse scraping:
   ```json
   {
     "cookies": {
       "_t": "your-t-token",
       "_forum_session": "your-forum-session"
     },
     "start_date": "2025-01-01",
     "end_date": "2025-04-15",
     "output_dir": "data/discourse",
     "category_id": 34,
   }
   ```

## Usage

### Scraping Data

1. **Course Content**:
   - Place course markdown files in `data/course`.
   - Analyze images in markdown files:
     ```bash
     uv run app/data course image-context data/course
     ```

2. **Discourse Posts**:
   - Scrape posts:
     ```bash
     uv run app/data discourse scrape config.json
     ```
   - Analyze images in posts:
     ```bash
     uv run app/data discourse image-context config.json
     ```

### Generating Embeddings

1. **Course Embeddings**:
   ```bash
   uv run app/embed course embed --input-dir data/course --output-file embeddings/course.npz
   ```

2. **Discourse Embeddings**:
   ```bash
   uv run app/embed discourse embed --input-dir data/discourse --output-file embeddings/discourse.npz
   ```

3. **Merge Embeddings**:
   ```bash
   uv run app/embed join-npz embeddings/course.npz embeddings/discourse.npz --output embeddings.npz
   ```

### Running the API

Start the FastAPI server:
```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Make a sample request:
```bash
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?\", \"image\": \"$(base64 -w0 project-tds-virtual-ta-q1.webp)\"}"
```

### Self-Updating CLI

Update `pyproject.toml` scripts:
```bash
uv run app self-update
```

## Evaluation

The application meets the project requirements:

- **Pre-requisites**:
  - Hosted in a public GitHub repository with an MIT `LICENSE` file.
  - API endpoint publicly accessible.

- **Functionality**:
  - Handles POST requests with questions and optional images.
  - Returns JSON responses with answers and links within 30 seconds.
  - Uses semantic search for relevant sources.

- **Bonus**:
  - Includes a Discourse scraping script (`app/data/discourse.py`) with date range support (+1 mark).
  - Designed for easy deployment, potentially suitable as an official solution (+2 marks).

To evaluate:
1. Update `project-tds-virtual-ta-promptfoo.yaml` with your API URL.
2. Run:
   ```bash
   npx -y promptfoo eval --config project-tds-virtual-ta-promptfoo.yaml
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.