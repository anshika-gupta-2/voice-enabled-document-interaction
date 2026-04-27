# multi-conversational-tool
# 🎙️ Voice-Enabled Document Chat System (RAG)

A Voice-Enabled Conversational AI system that allows users to upload documents and interact with them using natural language (text + voice). Built using Retrieval-Augmented Generation (RAG), LLMs, and real-time streaming for accurate, context-aware responses.

---

## 🚀 Features

- 📄 Upload and chat with documents (PDF, text, etc.)
- 🎙️ Voice input for queries
- 💬 Real-time streaming responses
- 🧠 Context-aware answers using RAG
- 🔎 Vector search using embeddings
- ⚡ FastAPI backend with WebSocket support
- 🔐 Modular and scalable architecture

---

---

## 🧠 How It Works (RAG Pipeline)

1. **Document Upload**
   - User uploads document (PDF/text)

2. **Preprocessing**
   - Text extraction
   - Chunking into smaller sections

3. **Embedding Generation**
   - Convert chunks into vector embeddings

4. **Vector Storage**
   - Store embeddings in Pinecone

5. **Query Processing**
   - User asks question (text/voice)

6. **Similarity Search**
   - Retrieve most relevant chunks

7. **LLM Response**
   - Pass context + query to LLM
   - Generate accurate answer

---
