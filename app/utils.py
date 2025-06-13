"""
Utility classes for caching, rule-based engine, and health checking
"""

import hashlib
import json
import re
import time
from collections import OrderedDict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import psutil

from .models import AnswerResponse, Link


class ResponseCache:
    """LRU-style in-memory cache for responses"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.access_times: Dict[str, datetime] = {}
        self.hits = 0
        self.misses = 0

    def _generate_key(self, question: str) -> str:
        """Generate MD5 hash key for question"""
        return hashlib.md5(question.lower().encode()).hexdigest()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.access_times:
            return True
        return datetime.now() - self.access_times[key] > timedelta(seconds=self.ttl)

    def _evict_expired(self):
        """Remove expired entries"""
        expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]

    def get(self, question: str) -> Optional[AnswerResponse]:
        """Get cached response for question"""
        key = self._generate_key(question)

        if key in self.cache and not self._is_expired(key):
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = datetime.now()
            self.hits += 1

            # Recreate AnswerResponse from cached data
            cached_data = self.cache[key]
            return AnswerResponse(**cached_data)

        self.misses += 1
        return None

    def put(self, question: str, response: AnswerResponse):
        """Cache response for question"""
        key = self._generate_key(question)

        # Convert response to dict for storage
        response_dict = response.model_dump()

        # Add to cache
        self.cache[key] = response_dict
        self.access_times[key] = datetime.now()

        # Move to end
        self.cache.move_to_end(key)

        # Evict if over limit
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        # Clean expired entries periodically
        if len(self.cache) % 100 == 0:
            self._evict_expired()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 3),
            "memory_usage_mb": round(
                len(json.dumps(list(self.cache.values()))) / 1024 / 1024, 2
            ),
        }


class RuleBasedEngine:
    """Rule-based knowledge matching engine"""

    def __init__(self):
        self.knowledge_base = {}
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load the TDS course knowledge base"""
        self.knowledge_base = {
            "ai_model": {
                "keywords": [
                    "gpt",
                    "ai model",
                    "chatgpt",
                    "openai",
                    "model policy",
                    "artificial intelligence",
                    "ai ethics",
                    "academic integrity",
                ],
                "answer": """
**GPT Model Policy for TDS Assignments:**

For Tools in Data Science assignments, here are the guidelines for using AI models like GPT:

🎯 **Allowed Uses:**
• Research and learning about concepts
• Getting explanations of programming concepts
• Debugging help and error explanation
• Code review and suggestions for improvement
• Understanding error messages and stack traces
• Learning best practices and design patterns

❌ **Not Allowed:**
• Direct copy-paste of AI-generated code as your solution
• Having AI write entire assignments for you
• Submitting AI-generated content without proper attribution
• Using AI to bypass learning objectives
• Cheating on exams or assessments using AI tools

✅ **Best Practices:**
• Always understand and modify AI-generated code
• Cite AI assistance in your submissions
• Use AI as a learning tool, not a replacement for thinking
• Focus on understanding concepts, not just getting answers
• Test and verify any AI-suggested solutions
• Document your learning process

📋 **Citation Format:**
When using AI assistance, include a note like:
"I used ChatGPT to help debug this function and understand the error message."

Remember: The goal is to learn data science tools effectively!
                """,
                "links": [
                    {
                        "url": "https://www.iitm.ac.in/academics/academic-integrity",
                        "text": "IIT Madras Academic Integrity Policy",
                    },
                    {
                        "url": "https://openai.com/policies/usage-policies",
                        "text": "OpenAI Usage Policies",
                    },
                    {
                        "url": "https://writing.mit.edu/wcc/ai",
                        "text": "MIT Guidelines on AI Usage",
                    },
                ],
                "topic": "ai_model_policy",
            },
            "fastapi": {
                "keywords": [
                    "fastapi",
                    "deploy",
                    "railway",
                    "deployment",
                    "api",
                    "web service",
                    "uvicorn",
                    "docker",
                    "hosting",
                ],
                "answer": """
**Deploying FastAPI to Railway:**

Railway is a great platform for deploying FastAPI applications! Here's a complete guide:

🚀 **Quick Setup Steps:**
1. Create a `Dockerfile` in your project root
2. Add a `railway.toml` configuration file (optional but recommended)
3. Connect your GitHub repository to Railway
4. Deploy automatically on git push

📁 **Required Files:**

**Dockerfile Example:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
```

**railway.toml (Optional):**
```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 10
restartPolicyType = "on_failure"
```

🔧 **Environment Variables:**
Set these in Railway dashboard:
• `PORT` (automatically set by Railway)
• `ENVIRONMENT=production`
• Any API keys or database URLs

💡 **Pro Tips:**
• Use environment variables for all configuration
• Implement health check endpoints (`/health`)
• Add proper logging for debugging
• Use Railway's built-in metrics monitoring
• Set up automatic deployments from main branch
• Test locally with Docker before deploying

🐛 **Common Issues:**
• Port binding: Always use `--host 0.0.0.0 --port $PORT`
• Missing dependencies: Pin exact versions in requirements.txt
• Static files: Use Railway's static file serving if needed
                """,
                "links": [
                    {
                        "url": "https://docs.railway.app/deploy/deployments",
                        "text": "Railway Deployment Guide",
                    },
                    {
                        "url": "https://fastapi.tiangolo.com/deployment/",
                        "text": "FastAPI Deployment Documentation",
                    },
                    {
                        "url": "https://docs.railway.app/reference/dockerfile",
                        "text": "Railway Dockerfile Reference",
                    },
                    {
                        "url": "https://docs.railway.app/deploy/builds",
                        "text": "Railway Build Process",
                    },
                ],
                "topic": "fastapi_deployment",
            },
            "pydantic": {
                "keywords": [
                    "pydantic",
                    "validation",
                    "model",
                    "schema",
                    "data validation",
                    "type hints",
                    "basemodel",
                    "field",
                ],
                "answer": """
**Pydantic Validation and Common Issues:**

Pydantic is essential for data validation in modern Python applications!

✅ **Key Concepts:**
• Automatic data validation using Python type hints
• JSON schema generation and serialization
• Environment variable management with pydantic-settings
• Custom validators and data transformation
• Integration with FastAPI for request/response validation

🐛 **Common Issues & Solutions:**

**1. Validation Errors:**
```python
# Problem: Field validation fails
# Solution: Use proper type hints and Field constraints
from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)  # ge = greater equal, le = less equal
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
```

**2. Optional Fields and Defaults:**
```python
from typing import Optional, List

class Config(BaseModel):
    debug: bool = False  # Default value
    api_key: Optional[str] = None  # Can be None
    tags: List[str] = []  # Empty list default
    timeout: float = Field(30.0, gt=0)  # Must be greater than 0
```

**3. Custom Validators:**
```python
from pydantic import validator, root_validator

class UserModel(BaseModel):
    email: str
    password: str
    confirm_password: str
    
    @validator('email')
    def email_must_be_valid(cls, v):
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @root_validator
    def passwords_match(cls, values):
        pw1, pw2 = values.get('password'), values.get('confirm_password')
        if pw1 is not None and pw2 is not None and pw1 != pw2:
            raise ValueError('Passwords do not match')
        return values
```

**4. Environment Variables with Settings:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    debug: bool = False
    api_key: str
    port: int = 8000
    
    class Config:
        env_file = ".env"
        env_prefix = "APP_"  # Variables like APP_DATABASE_URL
```

**5. Serialization and Aliases:**
```python
class APIResponse(BaseModel):
    user_id: int = Field(alias="userId")
    full_name: str = Field(alias="fullName")
    
    class Config:
        allow_population_by_field_name = True  # Accept both field name and alias
```

🎯 **Best Practices:**
• Always use type hints for automatic validation
• Provide meaningful error messages in validators
• Use Field() for additional constraints and documentation
• Test your validation logic thoroughly
• Use aliases for API compatibility
• Leverage pydantic-settings for configuration management
• Use dataclasses for simple data structures, Pydantic for validation

🔍 **Debugging Tips:**
• Use `model.dict()` to see serialized output
• Check `model.__fields__` to inspect field definitions
• Use `ValidationError` handling for user-friendly errors
                """,
                "links": [
                    {
                        "url": "https://docs.pydantic.dev/latest/",
                        "text": "Pydantic Documentation",
                    },
                    {
                        "url": "https://docs.pydantic.dev/latest/usage/validators/",
                        "text": "Pydantic Validators Guide",
                    },
                    {
                        "url": "https://docs.pydantic.dev/latest/usage/settings/",
                        "text": "Pydantic Settings Management",
                    },
                    {
                        "url": "https://docs.pydantic.dev/latest/usage/types/",
                        "text": "Pydantic Field Types",
                    },
                ],
                "topic": "pydantic_validation",
            },
            "pandas": {
                "keywords": [
                    "pandas",
                    "dataframe",
                    "data analysis",
                    "csv",
                    "data cleaning",
                    "groupby",
                    "merge",
                    "join",
                ],
                "answer": """
**Pandas for Data Science:**

Pandas is the cornerstone of data analysis in Python! Here are essential concepts:

📊 **Core Data Structures:**
• DataFrame: 2D labeled data structure (like Excel spreadsheet)
• Series: 1D labeled array (single column)

🔧 **Essential Operations:**

**1. Loading Data:**
```python
import pandas as pd

# Read various formats
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')
df = pd.read_sql(query, connection)
```

**2. Data Exploration:**
```python
df.head()  # First 5 rows
df.info()  # Data types and memory usage
df.describe()  # Statistical summary
df.shape  # (rows, columns)
df.columns  # Column names
df.dtypes  # Data types
```

**3. Data Cleaning:**
```python
# Handle missing values
df.dropna()  # Remove rows with NaN
df.fillna(value)  # Fill NaN with value
df.isnull().sum()  # Count missing values

# Remove duplicates
df.drop_duplicates()

# Data type conversion
df['column'] = df['column'].astype('int64')
```

**4. Data Selection:**
```python
# Select columns
df['column_name']  # Single column
df[['col1', 'col2']]  # Multiple columns

# Filter rows
df[df['column'] > 5]
df.query('column > 5 and other_col == "value"')
```

**5. GroupBy Operations:**
```python
# Group and aggregate
df.groupby('category').sum()
df.groupby(['cat1', 'cat2']).agg({
    'sales': 'sum',
    'quantity': 'mean'
})
```

💡 **Pro Tips:**
• Use `.loc[]` for label-based indexing
• Use `.iloc[]` for position-based indexing
• Chain operations for readable code
• Use `pd.concat()` instead of loops for combining DataFrames
• Always check data types after loading
                """,
                "links": [
                    {
                        "url": "https://pandas.pydata.org/docs/",
                        "text": "Pandas Documentation",
                    },
                    {
                        "url": "https://pandas.pydata.org/docs/getting_started/intro_tutorials/",
                        "text": "Pandas Tutorials",
                    },
                    {
                        "url": "https://pandas.pydata.org/docs/user_guide/10min.html",
                        "text": "10 Minutes to Pandas",
                    },
                ],
                "topic": "pandas_basics",
            },
            "git": {
                "keywords": [
                    "git",
                    "github",
                    "version control",
                    "commit",
                    "push",
                    "pull",
                    "branch",
                    "merge",
                    "clone",
                ],
                "answer": """
**Git Version Control for TDS Projects:**

Git is essential for managing your code and collaborating on projects!

🔧 **Basic Git Workflow:**

**1. Initial Setup:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**2. Starting a Repository:**
```bash
# Create new repo
git init
git add .
git commit -m "Initial commit"

# Clone existing repo
git clone https://github.com/username/repo.git
```

**3. Daily Workflow:**
```bash
# Check status
git status

# Add changes
git add file.py  # Single file
git add .        # All files

# Commit changes
git commit -m "Add new feature"

# Push to remote
git push origin main
```

**4. Branching:**
```bash
# Create and switch to new branch
git checkout -b feature-branch

# Switch between branches
git checkout main
git checkout feature-branch

# Merge branch
git checkout main
git merge feature-branch
```

**5. Collaboration:**
```bash
# Get latest changes
git pull origin main

# Handle conflicts
# Edit conflicted files
git add .
git commit -m "Resolve merge conflicts"
```

🎯 **Best Practices for TDS:**
• Make small, frequent commits with clear messages
• Use branches for different features or experiments
• Always pull before starting new work
• Don't commit large data files (use .gitignore)
• Include README.md with project setup instructions
• Use meaningful commit messages

📁 **.gitignore for Data Science:**
```
# Data files
*.csv
*.xlsx
data/

# Python
__pycache__/
*.pyc
.env

# Jupyter
.ipynb_checkpoints/

# Models
*.pkl
*.model
```

🚨 **Common Issues:**
• Merge conflicts: Edit files manually, then add and commit
• Large files: Use Git LFS for datasets
• Sensitive data: Never commit API keys or passwords
                """,
                "links": [
                    {"url": "https://git-scm.com/doc", "text": "Git Documentation"},
                    {
                        "url": "https://github.com/git-guides",
                        "text": "GitHub Git Guides",
                    },
                    {
                        "url": "https://learngitbranching.js.org/",
                        "text": "Interactive Git Tutorial",
                    },
                ],
                "topic": "git_basics",
            },
        }

    def match_question(self, question: str) -> Optional[Dict]:
        """Match question against knowledge base using keyword scoring"""
        question_lower = question.lower()

        best_match = None
        highest_score = 0

        for topic, data in self.knowledge_base.items():
            score = 0
            keyword_matches = 0

            for keyword in data["keywords"]:
                if keyword in question_lower:
                    keyword_matches += 1
                    # Weight longer keywords more heavily
                    score += len(keyword) * 2

                    # Bonus for exact keyword matches
                    if f" {keyword} " in f" {question_lower} ":
                        score += len(keyword)

            # Normalize score by number of keywords and add match bonus
            if keyword_matches > 0:
                normalized_score = (score / len(data["keywords"])) * (
                    1 + keyword_matches * 0.1
                )

                if normalized_score > highest_score:
                    highest_score = normalized_score
                    best_match = {
                        **data,
                        "topic": topic,
                        "confidence": min(0.95, normalized_score / 15),  # Cap at 95%
                        "matched_keywords": keyword_matches,
                        "raw_score": score,
                    }

        # Only return if confidence is reasonable (at least 10%)
        if best_match and best_match["confidence"] > 0.1:
            return best_match

        return None


class HealthChecker:
    """System health monitoring and diagnostics"""

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_error = None

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Get system metrics
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk_usage = psutil.disk_usage("/")
        except Exception as e:
            # Fallback if psutil fails
            memory_info = None
            cpu_percent = 0
            disk_usage = None

        health_data = {
            "uptime": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "capabilities": [
                "AI question answering via AIpipe",
                "Rule-based knowledge matching",
                "Response caching with LRU",
                "File attachment processing",
                "Health monitoring",
                "Error handling and fallbacks",
            ],
        }

        # Add system metrics if available
        if memory_info:
            health_data.update(
                {
                    "memory_usage": f"{memory_info.used / 1024 / 1024:.1f}MB",
                    "memory_percent": memory_info.percent,
                    "memory_available": f"{memory_info.available / 1024 / 1024:.1f}MB",
                }
            )

        if cpu_percent is not None:
            health_data["cpu_percent"] = cpu_percent

        if disk_usage:
            health_data.update(
                {
                    "disk_usage_percent": (disk_usage.used / disk_usage.total) * 100,
                    "disk_free": f"{disk_usage.free / 1024 / 1024 / 1024:.1f}GB",
                }
            )

        # Generate warnings
        warnings = self._get_warnings(
            memory_info.percent if memory_info else 0, cpu_percent if cpu_percent else 0
        )
        health_data["warnings"] = warnings

        # Add last error info if exists
        if self.last_error:
            health_data["last_error"] = {
                "message": str(self.last_error),
                "timestamp": getattr(self.last_error, "timestamp", "unknown"),
            }

        return health_data

    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human-readable form"""
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _get_warnings(self, memory_percent: float, cpu_percent: float) -> List[str]:
        """Generate system warnings based on metrics"""
        warnings = []

        if memory_percent > 85:
            warnings.append("High memory usage detected (>85%)")
        elif memory_percent > 70:
            warnings.append("Moderate memory usage (>70%)")

        if cpu_percent > 80:
            warnings.append("High CPU usage detected (>80%)")
        elif cpu_percent > 60:
            warnings.append("Moderate CPU usage (>60%)")

        if self.error_rate > 0.1:
            warnings.append(f"High error rate detected ({self.error_rate:.1%})")

        return warnings

    def increment_request_count(self):
        """Increment request counter"""
        self.request_count += 1

    def increment_error_count(self, error: Exception = None):
        """Increment error counter and store last error"""
        self.error_count += 1
        if error:
            self.last_error = error
            self.last_error.timestamp = datetime.now().isoformat()


class AIpipeClient:
    """Client for AIpipe API integration with comprehensive error handling"""

    def __init__(self, settings):
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=settings.api_timeout, headers={"User-Agent": "TDS-Virtual-TA/1.0.0"}
        )
        self.request_count = 0
        self.error_count = 0

    async def generate_response(
        self, question: str, files: Optional[List] = None
    ) -> Dict[str, Any]:
        """Generate response using AIpipe API with educational context"""

        self.request_count += 1

        # Prepare the prompt with TDS context
        system_prompt = """You are a helpful teaching assistant for the "Tools in Data Science" course at IIT Madras. 
        
        Your role is to:
        - Provide clear, educational answers focused on data science tools and programming
        - Include practical examples and code snippets when helpful
        - Be encouraging and supportive to students
        - Explain concepts step-by-step for better understanding
        - Suggest best practices and common pitfalls to avoid
        - Relate answers to real-world data science applications
        
        Keep your responses informative but concise, and always encourage learning and exploration."""

        user_prompt = f"Student question: {question}"

        # Add file context if files are provided
        if files:
            file_info = (
                f"\n\nNote: Student has attached {len(files)} file(s) to this question."
            )
            user_prompt += file_info

        # Prepare request payload according to AIpipe API format
        payload = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **self.settings.get_generation_params(),
        }

        try:
            response = await self.client.post(
                self.settings.aipipe_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            result = response.json()

            # Extract answer from response (handle different response formats)
            answer = self._extract_answer_from_response(result)

            # Generate relevant links based on question content
            links = self._generate_relevant_links(question)

            return {
                "answer": answer,
                "links": links,
                "model_name": self.settings.model_name,
                "metadata": {
                    "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                    "model_version": result.get("model", self.settings.model_name),
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "file_count": len(files) if files else 0,
                },
            }

        except httpx.TimeoutException:
            self.error_count += 1
            raise Exception("AIpipe request timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            self.error_count += 1
            error_detail = f"AIpipe API error (status {e.response.status_code})"
            if e.response.status_code == 429:
                error_detail += ": Rate limit exceeded. Please wait a moment."
            elif e.response.status_code >= 500:
                error_detail += ": Server error. Please try again later."
            raise Exception(error_detail)
        except Exception as e:
            self.error_count += 1
            raise Exception(f"Failed to generate AI response: {str(e)}")

    def _extract_answer_from_response(self, result: Dict) -> str:
        """Extract answer from various possible response formats"""
        # Try different possible response structures
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]

        # Try direct content field
        if "content" in result:
            return result["content"]

        # Try response field
        if "response" in result:
            return result["response"]

        # Fallback
        return "I couldn't generate a response at this time. Please try rephrasing your question."

    def _generate_relevant_links(self, question: str) -> List[Dict[str, str]]:
        """Generate relevant links based on question keywords"""
        question_lower = question.lower()
        links = []

        # Data Science Python Libraries
        if any(
            word in question_lower
            for word in ["python", "pandas", "numpy", "matplotlib", "seaborn"]
        ):
            links.append(
                {
                    "url": "https://pandas.pydata.org/docs/",
                    "text": "Pandas Documentation",
                }
            )

        if "numpy" in question_lower:
            links.append(
                {"url": "https://numpy.org/doc/", "text": "NumPy Documentation"}
            )

        if any(
            word in question_lower
            for word in ["matplotlib", "plot", "visualization", "graph"]
        ):
            links.append(
                {
                    "url": "https://matplotlib.org/stable/",
                    "text": "Matplotlib Documentation",
                }
            )

        # Web Development
        if "fastapi" in question_lower:
            links.append(
                {
                    "url": "https://fastapi.tiangolo.com/",
                    "text": "FastAPI Documentation",
                }
            )

        if any(
            word in question_lower
            for word in ["deploy", "deployment", "railway", "hosting"]
        ):
            links.append(
                {"url": "https://docs.railway.app/", "text": "Railway Documentation"}
            )

        # Version Control
        if any(word in question_lower for word in ["git", "github", "version control"]):
            links.append(
                {"url": "https://git-scm.com/doc", "text": "Git Documentation"}
            )

        # Data Validation
        if "pydantic" in question_lower:
            links.append(
                {
                    "url": "https://docs.pydantic.dev/latest/",
                    "text": "Pydantic Documentation",
                }
            )

        # Machine Learning
        if any(
            word in question_lower
            for word in ["scikit", "sklearn", "machine learning", "ml"]
        ):
            links.append(
                {
                    "url": "https://scikit-learn.org/stable/",
                    "text": "Scikit-learn Documentation",
                }
            )

        # Jupyter and Notebooks
        if any(word in question_lower for word in ["jupyter", "notebook", "ipynb"]):
            links.append(
                {
                    "url": "https://jupyter.org/documentation",
                    "text": "Jupyter Documentation",
                }
            )

        # Always include course resources
        links.append(
            {"url": "https://courses.iitm.ac.in", "text": "IIT Madras Course Portal"}
        )

        # Remove duplicates while preserving order
        seen_urls = set()
        unique_links = []
        for link in links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                unique_links.append(link)

        return unique_links

    async def check_health(self) -> Dict[str, Any]:
        """Check AIpipe service health with minimal request"""
        try:
            # Simple health check - try to generate a minimal response
            test_payload = {
                "model": self.settings.model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_new_tokens": 5,
                "temperature": 0.1,
            }

            start_time = time.time()
            response = await self.client.post(
                self.settings.aipipe_url,
                json=test_payload,
                timeout=10,  # Shorter timeout for health check
            )
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2),
                    "model": self.settings.model_name,
                    "total_requests": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1),
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "error": f"AIpipe service returned status {response.status_code}",
                    "response_time_ms": round(response_time, 2),
                }

        except httpx.TimeoutException:
            return {
                "status": "unhealthy",
                "error": "Health check timed out",
                "timeout_seconds": 10,
            }
        except httpx.HTTPStatusError as e:
            return {
                "status": "unhealthy",
                "status_code": e.response.status_code,
                "error": f"HTTP error: {e.response.status_code}",
            }
        except Exception as e:
            return {"status": "unhealthy", "error": f"Health check failed: {str(e)}"}

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - clean up HTTP client"""
        await self.client.aclose()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "success_rate": (self.request_count - self.error_count)
            / max(self.request_count, 1),
        }
