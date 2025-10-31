# 🧠 NutriGen: RAG-Based Nutritional Chatbot

NutriGen is a **Retrieval-Augmented Generation (RAG)** system designed to intelligently answer questions about **nutrition and dietary science** using contextual retrieval, embeddings, and LLM-based augmentation.  
It combines **Jupyter-based data preparation**, **Supabase vector search**, and a **Next.js + OpenAI** frontend for real-time conversational responses.

---

## 🚀 Overview

NutriGen leverages the power of:
- **LLM embeddings & retrieval** for understanding nutrition documents
- **Supabase pgvector** for fast similarity search
- **OpenAI GPT models** (and optionally Gemma 7B locally) for natural language responses
- **Next.js frontend** for an elegant chat-based user interface

The system reads and embeds research documents, stores them as vectors in Supabase, and then uses OpenAI to generate context-aware responses.

---

## 🧩 System Architecture

📁 NutriGen/
├── 📓 main_notebook.ipynb ← Jupyter notebook for manual chunking & embedding
│ ├─ Handles 5 chunking strategies
│ ├─ Embedding via all-mpnet-base-v2 and local Gemma 7B
│ └─ Performs RAG pipeline testing locally
│
├── 📁 Backend API/
│ ├── ingest.py ← Automated ingestion + embedding pipeline
│ │ ├─ Uses OpenAI text-embedding-3-small
│ │ ├─ Stores embeddings in Supabase (pgvector)
│ │ └─ Connects PostgreSQL for vector queries
│ └── ... other scripts
│
├── 📁 app/
│ ├── api/chat/route.ts ← Next.js API route for RAG query + OpenAI augmentation
│ └── page.tsx ← Frontend chat interface
│
├── 📄 .env ← Environment variables (OpenAI & Supabase keys)
├── 📄 package.json
├── 📄 README.md
└── 📁 public/
└── (images for visualization)

yaml
Copy code

---

## 🧠 Workflow

### 1️⃣ Data Preparation (in Jupyter)
- Manually chunked **5 types of text splitting** for different structural analysis.
- Used **Sentence Transformers (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`)** to embed text.
- Tested **Gemma 7B** locally for comparison in generation quality.
- Stored embeddings temporarily for inspection and visualization.

### 2️⃣ Backend Ingestion (Supabase + pgvector)
- `ingest.py` automates embedding creation and uploading.
- Uses OpenAI’s `text-embedding-3-small` model.
- Embeddings are stored in **Supabase (PostgreSQL + pgvector)**.
- A **RPC function `match_documents`** in Supabase retrieves top-k matches by cosine similarity.

### 3️⃣ RAG API (Next.js Route)
- `route.ts` receives a query → embeds it → retrieves relevant chunks via Supabase RPC.
- If results found: constructs a contextual prompt and sends to **OpenAI Chat Completion API**.
- If not: returns a helpful fallback response.

### 4️⃣ Frontend (Next.js + Tailwind + Framer Motion)
- Built in `/app/page.tsx`
- Features a **minimal sleek chat interface** inspired by OpenAI Chat UI.
- Displays:
  - User + Assistant messages
  - Animated typing indicator (three dots while generating)
  - **Clickable citations** showing contextual snippets from source pages
  - Dynamic sources section with similarity scores
- Styled in the **theme of [arcprize.org](https://arcprize.org/arc-agi)** for a refined scientific aesthetic.

---

## 🧰 Tech Stack

| Layer | Tools / Frameworks |
|-------|--------------------|
| **Frontend** | Next.js 15, React, Tailwind CSS, Framer Motion |
| **Backend API** | Node.js, OpenAI API |
| **Database / Vector Store** | Supabase (PostgreSQL + pgvector) |
| **Embedding Models** | OpenAI `text-embedding-3-small`, Sentence Transformers |
| **Generation Models** | OpenAI GPT-4o / GPT-3.5, Gemma 7B (local) |
| **Data Prep Notebook** | Python, torch, sentence-transformers, numpy, pandas |

---



<img width="1280" height="800" alt="demo1" src="https://github.com/user-attachments/assets/6935b4ca-2a23-4dba-aba3-ec47c4f2f3f8" />

<img width="1280" height="800" alt="dem2" src="https://github.com/user-attachments/assets/ea3547ce-a077-4899-98b7-c74ae67991b6" />
<img width="1280" height="800" alt="dem3" src="https://github.com/user-attachments/assets/aadd8164-941e-4ada-8cf0-b2ee9cabf52d" />



## ⚙️ Environment Variables

Create a `.env` file in the root:
```bash
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
