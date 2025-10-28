# Quick Start Guide

Get the Moffitt Agentic RAG System running in **under 5 minutes**!

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Groq API key** (free) or **OpenAI API key** (paid)

## Steps

### 1. Clone & Navigate
```bash
git clone <repository-url>
cd moffitt-agentic-rag
```

### 2. Setup Environment
```bash
# Copy the example file
cp .env.docker.example .env

# Edit .env and add your API key
# Change: GROQ_API_KEY=your_groq_api_key_here
# To:     GROQ_API_KEY=gsk_abc123...
```

**Get a free Groq API key**: https://console.groq.com/keys

### 3. Start Everything
```bash
docker-compose up --build
```

Wait 1-2 minutes for:
- Dependencies to install
- Models to download
- Services to start

### 4. Access the App

- **Frontend UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

### 5. Test It

Try a query in the UI:
> "Who studies cancer evolution?"

---

## Troubleshooting

### Problem: "API key not found"
**Solution**: Make sure you edited `.env` and replaced `your_groq_api_key_here` with your actual key

### Problem: "Port already in use"
**Solution**: Stop other services using ports 3000 or 8000, or change ports in `docker-compose.yml`

### Problem: Build fails
**Solution**:
```bash
docker system prune -a
docker-compose build --no-cache
```

---

## Next Steps

- See [SETUP.md](SETUP.md) for detailed setup options
- See [README.md](README.md) for architecture overview
- See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development guide

---

## Configuration Options

All settings in `.env` with defaults:

```bash
# Required: Set ONE of these
GROQ_API_KEY=your_key_here          # Free tier available
OPENAI_API_KEY=your_key_here        # Paid

# Optional: Choose provider (default: groq)
LLM_PROVIDER=groq                   # or: openai, ollama

# Optional: Choose models (sensible defaults set)
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Optional: Authentication (default: dev_api_key)
API_KEY=dev_api_key
```

That's it! 🎉
