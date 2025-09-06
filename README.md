SmartDoc Assistant – RAG-based Chatbot with LINE Integration

A lightweight chatbot that can read and understand PDF documents using RAG (Retrieval-Augmented Generation), powered by Sentence Transformers + ChromaDB + LLM, and integrated into LINE Messaging Platform for real-time interaction.

📄 Demo PDF: 5-dimensions-happiness.pdf (97 pages, 2292 lines)

Features
- Extracts & chunks PDF content with overlapping context
- Embeds chunks with paraphrase-multilingual-MiniLM-L12-v2
- Stores vector index with ChromaDB
- Answers user questions using similarity search
- Sends intelligent replies via LINE chatbot
- Supports multilingual queries (Thai & English)