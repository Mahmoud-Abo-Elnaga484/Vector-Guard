# 🦟 VectorGuard AI

> **Evidence-grounded differential diagnosis for mosquito-borne viral diseases.**

VectorGuard AI is an advanced, medically-aligned Retrieval-Augmented Generation (RAG) pipeline designed to assist with conversational clinical information regarding mosquito-transmitted viral illnesses (Dengue, Zika, Chikungunya, and Yellow Fever). 

Unlike standard conversational AI, VectorGuard is strictly constrained to WHO clinical guidelines. It guarantees zero-hallucination citations by structurally enforcing the LLM to output only valid chunk IDs, backed by a real-time LLM-as-a-judge evaluation system.

---

## ✨ Core Features

*   **🏥 Clinical Focus:** Specialized in WHO guidelines for Dengue Fever, Chikungunya, Yellow Fever, and Zika Virus.
*   **🛡️ Strict Guardrails:** Automatically refuses out-of-scope medical questions (e.g., general cardiology) or dangerous queries before they even reach the generation model.
*   **🧩 Structurally Safe Citations:** The generation model is restricted to outputting `chunk_ids` only. The backend resolves these IDs to actual documents and pages, making fabricated citations structurally impossible.
*   **📊 Live Evaluation Metrics:** Displays real-time **Faithfulness** and **Citation Accuracy** scores directly in the UI for every response using an isolated LLM Judge.
*   **🔍 Advanced RAG Architecture:** Utilizes Section-Aware and Recursive Chunking to maintain clinical context, ensuring symptoms are accurately mapped to their respective diseases.

---

## 🛠️ Tech Stack

*   **Frontend UI:** [Streamlit](https://streamlit.io/)
*   **LLM Provider (Generation):** Google Gemini (`gemini-3.5-flash`) via OpenAI-compatible endpoint.
*   **LLM Provider (Evaluation/Judge):** Google Gemini (`gemini-3.5-flash-lite`).
*   **Embeddings:** `BAAI/bge-small-en-v1.5`
*   **Vector Database:** [Qdrant](https://qdrant.tech/)
*   **Document Processing:** LlamaCloud Parse

---

## 🏗️ Architecture / Pipeline

1.  **Ingestion:** Medical documents are parsed and split using Section-Aware Recursive Chunking (`Chunk Size: 2000`, `Overlap: 200`). This preserves document headers (e.g., *Dengue -> Treatment*) so the LLM doesn't lose context.
2.  **Retrieval:** Queries are embedded using `bge-small-en-v1.5` (with proper search prefixes) to fetch the Top-K relevant chunks from Qdrant. A hard threshold is applied; if relevance is below 0.30, generation is blocked.
3.  **Generation:** The `gemini-3.5-flash` model synthesizes the clinical response and returns raw JSON containing only validated `chunk_ids`.
4.  **Evaluation:** A separate `gemini-3.5-flash-lite` judge evaluates the output for logical entailment, measuring *Retrieval Precision*, *Citation Accuracy*, and *Faithfulness*.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key
*   LlamaCloud API Key
