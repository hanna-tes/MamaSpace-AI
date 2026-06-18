
# 🤱 MamaSpace AI: A Guardrailed GenAI RAG for Maternal Mental Wellbeing

> *Built with deep personal passion and clinical rigor to provide safe, empathetic, and immediate support for mothers navigating postpartum mental health challenges.*

## 🌸 The Mission
The postpartum period can be incredibly isolating. As a mother who has personally experienced postpartum anxiety, I built **MamaSpace** to bridge the gap between clinical mental health resources and a mother's immediate need for empathetic support. 

While designed for maternal mental health, this architecture is highly scalable and directly applicable to supporting vulnerable young women and adolescent mothers in the Global South, aligning with organizations dedicated to scaling safe, AI-augmented behavioral interventions.

## 🛡️ Safety & Guardrails (Core Feature)
When dealing with vulnerable users, AI safety is not an afterthought; it is the foundation. MamaSpace implements strict, multi-layered guardrails to ensure clinical safety:
*   **Deterministic Crisis Escalation:** If a user expresses thoughts of self-harm or harm to the baby, the LLM is completely bypassed. The system instantly triggers a hardcoded, empathetic crisis response with emergency hotline numbers.
*   **Medical Advice Blockade:** The bot is strictly prohibited from diagnosing or prescribing (especially regarding breastfeeding and medications). It gently redirects users to their OB-GYN or healthcare providers.
*   **Clinical RAG Pipeline:** The AI does not guess. It retrieves answers exclusively from verified, peer-reviewed medical institutions (NHS, March of Dimes, Cleveland Clinic).

## 🧠 Technical Architecture
*   **Orchestration & RAG:** Python, LangChain
*   **Vector Database:** PostgreSQL with `pgvector` extension (Enterprise-grade, scalable infrastructure)
*   **LLM Inference:** Groq (Llama 3) for lightning-fast, empathetic conversational generation
*   **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Frontend:** Streamlit (Custom CSS for a warm, low-cognitive-load "mother-baby" aesthetic)

## 📚 Clinical Knowledge Base
The RAG vector database is populated exclusively with trusted, open-access clinical guidelines:
1.  **NHS (UK):** Postnatal depression symptoms and treatments.
2.  **March of Dimes:** Postpartum mental health, PPD, PPA, and PTSD guidelines.
3.  **Cleveland Clinic:** Comprehensive PPD causes, risk factors, and coping strategies.
4.  **Hackensack Meridian Health:** Guides on supporting a loved one with postpartum anxiety/depression.

## 🚀 How to Run Locally

### Prerequisites
*   Python 3.9+
*   PostgreSQL installed locally with the `pgvector` extension enabled.
*   Groq API key.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hanna-tes/MamaSpace-AI.git
   cd MamaSpace-AI
