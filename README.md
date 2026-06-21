# 🤱 MamaSpace AI: A Guardrailed GenAI RAG for Maternal Mental Wellbeing

Built with deep personal passion and clinical rigor to provide safe, empathetic, and immediate support for mothers navigating postpartum mental health challenges.

## 🌸 The Mission
The postpartum period can be incredibly isolating. As a mother who has personally experienced postpartum anxiety, I built MamaSpace to bridge the gap between clinical mental health resources and a mother's immediate need for empathetic support.

While designed for maternal mental health, this architecture is highly scalable and directly applicable to supporting vulnerable young women and adolescent mothers in the Global South, aligning with organizations dedicated to scaling safe, AI-augmented behavioral interventions.

**Accessibility First:** MamaSpace is deployed on **both web and Telegram**. This ensures support is available regardless of device type, internet reliability, or digital literacy level.

## ️ Safety & Guardrails (Core Feature)
When dealing with vulnerable users, AI safety is not an afterthought; it is the foundation. MamaSpace implements strict, multi-layered guardrails to ensure clinical safety:

-   **Deterministic Crisis Escalation:** If a user expresses thoughts of self-harm or harm to the baby (in English OR Amharic), the LLM is completely bypassed. The system instantly triggers a hardcoded, empathetic crisis response with Ethiopia-specific emergency hotline numbers (908/991).
-   **Medical Advice Blockade:** The bot is strictly prohibited from diagnosing or prescribing (especially regarding breastfeeding and medications). It gently redirects users to their OB-GYN or healthcare providers.
-   **Clinical RAG Pipeline:** The AI does not guess. It retrieves answers exclusively from verified, peer-reviewed medical institutions (NHS, March of Dimes, Cleveland Clinic).
-   **Bilingual Safety Keywords:** Custom Amharic keyword detection ensures crisis signals are never missed due to language barriers.

##  Technical Architecture

### Multi-Platform Deployment
| Platform | Tech Stack | Use Case |
| :--- | :--- | :--- |
| **Web App** | Streamlit + Custom CSS | Rich visual experience for broadband users |
| **Telegram Bot** | python-telegram-bot v21 (Async) | Low-data, accessible interface for mobile-first users |
| **Shared Logic** | Decoupled `shared_logic.py` | Identical RAG/Safety engine across both platforms |

### Core Infrastructure
-   **Orchestration & RAG:** Python, LangChain
-   **Vector Database:** PostgreSQL with pgvector extension (Neon DB, scalable)
-   **LLM Inference:** Groq (Llama-3.3-70B) for lightning-fast, multilingual conversational generation
-   **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
-   **Frontend:** Streamlit (Custom pink gradient theme for low-cognitive-load "mother-baby" aesthetic)
-   **Deployment:** Render Free Tier (Zero-cost, auto-scaling)

## 📚 Clinical Knowledge Base
The RAG vector database is populated exclusively with trusted, open-access clinical guidelines:

-   **NHS (UK):** Postnatal depression symptoms and treatments.
-   **March of Dimes:** Postpartum mental health, PPD, PPA, and PTSD guidelines.
-   **Cleveland Clinic:** Comprehensive PPD causes, risk factors, and coping strategies.
-   **Hackensack Meridian Health:** Guides on supporting a loved one with postpartum anxiety/depression.

## 🇪🇹 Amharic Language Support
MamaSpace natively supports Amharic (አማርኛ) through:
-   Few-shot prompting with culturally appropriate validation phrases
-   Structured response templates ensuring grammatical correctness
-   Zero-temperature LLM settings to prevent linguistic hallucinations
-   Bilingual crisis responses with local Ethiopian resources

## 🚀 How to Run Locally

### Prerequisites
-   Python 3.9+
-   PostgreSQL installed locally with the `pgvector` extension enabled
-   Groq API key
-   Telegram Bot Token (from @BotFather)

### Installation
```bash
git clone https://github.com/hanna-tes/MamaSpace-AI.git
cd MamaSpace-AI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
