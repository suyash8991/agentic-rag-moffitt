# Moffitt Agentic RAG System - Setup Guide

This guide will help you set up and run the Moffitt Agentic RAG System anywhere - whether on Windows, macOS, or Linux.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start with Docker](#quick-start-with-docker)
- [Local Development Setup](#local-development-setup)
- [Environment Configuration](#environment-configuration)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Prerequisites

### For Docker Setup (Recommended)
- **Docker** version 20.10 or higher
- **Docker Compose** version 2.0 or higher
- **Git** (to clone the repository)
- **API Keys** (at least one):
  - Groq API key ([Get it free](https://console.groq.com/keys))
  - OpenAI API key (optional, [Get it here](https://platform.openai.com/account/api-keys))

### For Local Development
- **Python 3.10 or higher**
- **Node.js 16 or higher** with npm
- **Git**
- **API Keys** (same as above)

---

## Quick Start with Docker

This is the **easiest** way to run the application anywhere.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd moffitt-agentic-rag
```

### Step 2: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.docker.example .env
   ```

2. Edit `.env` and add your API key (at minimum, set one of these):
   ```bash
   # For Groq (Free, recommended)
   GROQ_API_KEY=your_actual_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile

   # OR for OpenAI (Paid)
   # OPENAI_API_KEY=your_actual_openai_api_key_here
   # OPENAI_MODEL=gpt-4o-mini
   # LLM_PROVIDER=openai
   ```

   **Note**: All other settings have sensible defaults. You can customize models, embedding settings, and more. See the [Environment Configuration](#environment-configuration) section for details.

### Step 3: Start the Application

```bash
docker-compose up --build
```

This will:
- Build the backend and frontend containers
- Install all dependencies automatically
- Start both services with proper networking
- Make the app available at:
  - **Frontend**: http://localhost:3000
  - **Backend API**: http://localhost:8000
  - **API Docs**: http://localhost:8000/docs

### Step 4: Verify Everything Works

Open http://localhost:3000 in your browser and try a query like:
> "Who studies cancer evolution?"

---

## Local Development Setup

If you prefer to run without Docker or want to develop locally.

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   # From the backend directory
   cp .env.example .env
   # Edit .env and add your API keys (at minimum GROQ_API_KEY or OPENAI_API_KEY)
   ```

   Or use the root-level `.env.example`:
   ```bash
   # From project root
   cd ..
   cp .env.example backend/.env
   # Edit backend/.env and add your API keys
   ```

5. **Validate setup**:
   ```bash
   python validate_env.py
   ```

6. **Start the backend**:
   ```bash
   uvicorn main:app --reload
   ```

   Backend will be available at http://localhost:8000

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment**:
   ```bash
   # Create .env file
   echo "REACT_APP_API_URL=http://localhost:8000" > .env
   echo "REACT_APP_API_KEY=dev_api_key" >> .env
   ```

4. **Start the frontend**:
   ```bash
   npm start
   ```

   Frontend will open automatically at http://localhost:3000

---

## Environment Configuration

### Required Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_PROVIDER` | LLM provider to use (`groq` or `openai`) | `groq` | Yes |
| `GROQ_API_KEY` | Groq API key | - | If using Groq |
| `OPENAI_API_KEY` | OpenAI API key | - | If using OpenAI |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API key for backend authentication | `dev_api_key` |
| `GROQ_MODEL` | Groq model to use | `llama-3.3-70b-versatile` |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `EURON_API_KEY` | Euron API key (optional) | - |
| `EURON_MODEL` | Euron model to use | `gpt-4.1-nano` |
| `LLM_MODEL_NAME` | Ollama model name | `llama3` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `EMBEDDING_MODEL_NAME` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `VECTOR_DB_DIR` | Path to vector database | `../data/vector_db` |
| `PROCESSED_DATA_DIR` | Path to processed data | `../data/processed` |

### Getting API Keys

#### Groq API Key (Free, Recommended)
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to **API Keys** section
4. Click **Create API Key**
5. Copy the key and paste it in your `.env` file

#### OpenAI API Key (Paid)
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **Create new secret key**
5. Copy the key and paste it in your `.env` file

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors

**Problem**: Python packages not installed correctly

**Solution**:
```bash
cd backend
pip install -r requirements.txt
```

#### 2. "API key not found" or "API key invalid"

**Problem**: Environment variables not loaded or invalid key

**Solution**:
1. Check your `.env` file exists
2. Verify API key is correct (no quotes, no extra spaces)
3. Restart the application after changing `.env`
4. Run validation: `python validate_env.py`

#### 3. "Address already in use" error

**Problem**: Port 8000 or 3000 is already in use

**Solution**:
```bash
# Find and kill process using the port
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

Or change the port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change 8001 to any available port
```

#### 4. Docker build fails

**Problem**: Network issues or disk space

**Solution**:
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

#### 5. Vector database not found

**Problem**: Data directories don't exist

**Solution**:
```bash
# Create required directories
mkdir -p data/vector_db
mkdir -p data/processed
mkdir -p data/markdown
```

#### 6. CORS errors in browser

**Problem**: Frontend can't connect to backend

**Solution**:
1. Check backend is running: `curl http://localhost:8000/api/health`
2. Verify `CORS_ORIGINS` includes your frontend URL
3. Check browser console for specific error

#### 7. Model download fails

**Problem**: HuggingFace model download timeout

**Solution**:
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

---

## Verification

### 1. Backend Health Check

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "provider": "groq"
}
```

### 2. Frontend Loading

Open http://localhost:3000 - you should see the chat interface

### 3. End-to-End Test

1. Go to http://localhost:3000
2. Enter a query: "Who studies cancer evolution?"
3. You should receive a response from the agent

### 4. Run Validation Script (Local setup only)

```bash
cd backend
python validate_env.py
```

All checks should pass:
```
✓ Python 3.10+ ✓
✓ .env file exists ✓
✓ Environment Variables ✓
✓ Data Directories ✓
✓ Python Dependencies ✓
```

---

## Additional Resources

- **API Documentation**: http://localhost:8000/docs (when backend is running)
- **Development Guide**: [DEVELOPMENT.md](docs/DEVELOPMENT.md)
- **Tools Documentation**: [TOOLS.md](docs/TOOLS.md)
- **Main README**: [README.md](README.md)

---

## Production Deployment

For production deployment, see the deployment guide in `docs/DEPLOYMENT.md` (coming soon).

Key considerations for production:
- Use production-ready API keys
- Set `API_KEY` to a secure random value
- Use environment-specific `.env` files
- Enable HTTPS
- Set up proper logging and monitoring
- Consider using a managed vector database

---

## Support

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review backend logs: `docker-compose logs backend`
3. Review frontend logs: `docker-compose logs frontend`
4. Run the validation script: `python backend/validate_env.py`
5. Open an issue on GitHub with:
   - Your operating system
   - Python/Node version
   - Error messages
   - Steps to reproduce

---

## Success Checklist

Before considering your setup complete, verify:

- [ ] Docker containers are running (if using Docker)
- [ ] Backend health check returns "healthy"
- [ ] Frontend loads at http://localhost:3000
- [ ] API documentation loads at http://localhost:8000/docs
- [ ] You can submit a query and receive a response
- [ ] No errors in browser console
- [ ] No errors in backend logs

If all checks pass, congratulations! Your Moffitt Agentic RAG System is ready to use! 🎉
