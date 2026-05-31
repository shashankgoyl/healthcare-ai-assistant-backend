# 🏥 Healthcare AI Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** AI assistant that answers healthcare questions from a curated set of clinical and operational documents.

**Backend** → Python (FastAPI) — deploy on **Koyeb**  
**Frontend** → React + Vite — deploy on **Netlify**

---

## 📐 Architecture

```
User Question
     │
     ▼
┌             ┐
│              FastAPI Backend               │
│                                            │
│  POST /ask                                 │
│     │                                      │
│     ▼                                      │
│  ┌           ┐   │
│  │          Agent Router               │   │
│  │  detect_intent(question)            │   │
│  └   ┬      ┬    ┘   │
│         │ appointment      │ rag            │
│         ▼                  ▼               │
│  ┌     ┐   ┌      ┐     │
│  │ Mock Tool  │   │   RAG Pipeline   │     │
│  │ check_     │   │                  │     │
│  │ available_ │   │ 1. Embed query   │     │
│  │ slots()    │   │ 2. ChromaDB      │     │
│  └     ┘   │    similarity    │     │
│                   │    search        │     │
│                   │ 3. Groq LLM      │     │
│                   │    generation    │     │
│                   └      ┘     │
└             ┘
                   │
                   ▼
          Structured JSON Response
          { answer, sources, confidence }
```

---

## 🗂️ Project Structure

```
healthcare-ai-assistant/
├  app/
│   ├  main.py          # FastAPI app, endpoints
│   ├  rag.py           # Document ingestion, chunking, retrieval
│   ├  embeddings.py    # sentence-transformers wrapper
│   ├  llm.py           # Groq LLM integration + system prompt
│   ├  agent.py         # Intent router + appointment mock tool
│   └  config.py        # All config via env vars
├  data/
│   ├  discharge_instructions.txt
│   ├  appointment_scheduling_policy.txt
│   ├  insurance_eligibility_faq.txt
│   ├  hipaa_privacy_guidelines.txt
│   ├  medication_refill_policy.txt
│   └  telehealth_guidelines.txt
├  vector_store/        # ChromaDB persisted index
├  tests/
│   └  test_app.py      # Unit + integration tests
├  frontend/
│   ├  src/
│   │   ├  App.jsx
│   │   ├  pages/
│   │   │   ├  ChatPage.jsx
│   │   │   └  AdminPage.jsx
│   │   ├  components/
│   │   │   ├  Header.jsx
│   │   │   ├  ChatMessage.jsx
│   │   │   ├  ChatInput.jsx
│   │   │   └  SuggestedQuestions.jsx
│   │   ├  hooks/
│   │   │   └  useChat.js
│   │   └  utils/
│   │       └  api.js
│   ├  package.json
│   ├  vite.config.js
│   └  .env.example
├  requirements.txt
├  Dockerfile
├  docker-compose.yml
├  .env.example
└  README.md
```

---

## ⚙️ Technical Choices

| Component | Choice | Reason |
|---|---|---|
| **LLM** | Groq API — `llama3-8b-8192` | Fast inference, free tier, OpenAI-compatible |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Small (80MB), fast, strong semantic quality |
| **Vector DB** | ChromaDB (persistent) | Simple, file-based, no external service needed |
| **API** | FastAPI | Async, fast, auto-docs at `/docs` |
| **Frontend** | React + Vite | Fast build, simple deployment to Netlify |

---

## 🔑 System Prompt

The following prompt is used for all RAG answers:

```
You are a knowledgeable and professional healthcare information assistant.

Your role is to answer questions using ONLY the context passages provided to you.

STRICT RULES you must always follow:
1. Base your answer EXCLUSIVELY on the provided context. Do not use any outside knowledge.
2. If the context does not contain the information needed to answer the question, respond exactly with:
   "I could not find this information in the provided documents."
3. Never guess, speculate, or extrapolate beyond what is explicitly stated in the context.
4. Do not provide direct medical diagnoses, prescribe treatments, or give personalised medical advice.
5. When appropriate, advise the user to consult a qualified healthcare professional.
6. Keep responses clear, concise, and professional.
7. If the question is partially answered by the context, share what you found and note the limitation.
8. Cite document names when referencing specific policies or guidelines.

You are here to help users understand healthcare policies and guidelines — not to replace a doctor.
```

---

## 🤖 Agent Workflow

The agent uses keyword-based intent detection to route questions:

```
question → detect_intent()
               │
   ┌    ┴    ┐
   │ "appointment"            │ "rag"
   ▼                          ▼
check_available_slots()   RAG pipeline
(mock tool)               (ChromaDB + Groq)
```

**Appointment keywords**: book, schedule, appointment, slot, available, cardiology, orthopedics, etc.

**Appointment tool** returns mock availability for 7 departments over the next 7 days.

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/healthcare-ai-assistant.git
cd healthcare-ai-assistant
```

### 2. Backend setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com/

### 3. Ingest documents

```bash
# Start the API server
uvicorn app.main:app --reload --port 8000

# In another terminal, ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can a patient request a medication refill through telehealth?"}'

# Appointment question (routes to mock tool)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I book a cardiology appointment for Monday?"}'
```

### 5. Frontend setup

```bash
cd frontend
cp .env.example .env.local
# Set VITE_API_URL=http://localhost:8000

npm install
npm run dev
# Open http://localhost:5173
```

---

## 🐳 Docker

### Build and run with Docker

```bash
# Build image
docker build -t healthcare-ai-assistant .

# Run container
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  -v $(pwd)/vector_store:/app/vector_store \
  healthcare-ai-assistant
```

### Docker Compose (recommended)

```bash
# Copy and edit .env
cp .env.example .env
# Set GROQ_API_KEY in .env

# Start
docker compose up --build

# Stop
docker compose down
```

After starting, ingest documents:
```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'
```

---

## 🚢 Deployment

### Backend → Koyeb

1. Push your code to GitHub.
2. Go to [koyeb.com](https://app.koyeb.com) → **Create Service** → **GitHub**.
3. Select your repository.
4. Set **Build command**: `pip install -r requirements.txt`
5. Set **Run command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
6. Add environment variables:
   - `GROQ_API_KEY` = your Groq key
   - `ALLOWED_ORIGINS` = `https://your-app.netlify.app`
7. After deploy, call `POST /ingest` once to populate the vector store.

> **Note**: On Koyeb free tier, the filesystem is ephemeral. For persistence, mount a volume or use a hosted vector DB (e.g. Pinecone, Weaviate Cloud).

### Frontend → Netlify

1. Push your code to GitHub.
2. Go to [netlify.com](https://app.netlify.com) → **Add new site** → **Import from Git**.
3. Select your repository and set:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
4. Add environment variable:
   - `VITE_API_URL` = `https://your-backend.koyeb.app`
5. Deploy!

---

## 🧪 Running Tests

```bash
# From project root with venv activated
pytest tests/ -v

# With coverage
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 📡 API Reference

### `GET /health`
Returns system health and vector store stats.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "collection_stats": {
    "collection_name": "healthcare_docs",
    "total_chunks": 124
  }
}
```

### `POST /ingest`
Ingest documents from the data folder.

```json
// Request (body optional)
{ "data_dir": null }

// Response
{
  "status": "success",
  "files_processed": 6,
  "chunks_stored": 124,
  "collection_count": 124,
  "errors": [],
  "message": "Successfully ingested 6 file(s) into 124 chunks."
}
```

### `POST /ask`
Answer a healthcare question.

```json
// Request
{ "question": "Can a patient request a medication refill through telehealth?" }

// Response
{
  "answer": "Yes, patients can request medication refills through telehealth...",
  "sources": [
    {
      "document": "medication_refill_policy.txt",
      "chunk": "Medication refill requests may be reviewed during telehealth visits..."
    }
  ],
  "confidence": "high",
  "intent": "rag",
  "tool_used": "rag_pipeline",
  "processing_time_ms": 1243.5
}
```

---

## 💬 Sample Questions & Expected Responses

| Question | Tool | Expected |
|---|---|---|
| "Can a patient request a medication refill through telehealth?" | RAG | Yes, with conditions per telehealth policy |
| "What are my rights under HIPAA?" | RAG | Lists patient rights from HIPAA document |
| "What happens if I miss an appointment?" | RAG | Late cancellation/no-show fee policy |
| "Can I book a cardiology appointment for Monday?" | Appointment Tool | Mock available slots for Cardiology |
| "What is prior authorization?" | RAG | Insurance FAQ explanation |
| "When should I go to the ER after discharge?" | RAG | Emergency symptoms list from discharge doc |
| "What is the weather today?" | RAG | "I could not find this information..." |

---

## ⚠️ Limitations & Future Improvements

**Current Limitations:**
- Vector store is file-based (not suitable for high-concurrency or serverless)
- Appointment tool is mock only (no real scheduling system)
- No user authentication or session management
- Single-turn Q&A only (no conversation memory)
- Embedding model runs locally (slow cold start on serverless)

**Future Improvements:**
- [ ] Persistent cloud vector DB (Pinecone / Weaviate Cloud)
- [ ] Streaming LLM responses via SSE
- [ ] Multi-turn conversation with history
- [ ] User authentication (Auth0, Clerk)
- [ ] Document upload UI with drag-and-drop
- [ ] Real appointment booking integration (HealthKit, Epic FHIR)
- [ ] Answer evaluation / feedback loop
- [ ] Multi-language support
- [ ] Hybrid search (BM25 + semantic)
- [ ] Reranking with a cross-encoder model

---

## 📄 License

MIT — free to use, modify, and distribute.
