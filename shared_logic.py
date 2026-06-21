# shared_logic.py
import os
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ==========================================
# SAFETY GUARDRAILS
# ==========================================
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "hurt myself", "harm myself", 
    "hurt the baby", "harm the baby", "end my life", 
    "postpartum psychosis", "hallucinations", "want to die",
    "ራሴን ማጥፋት", "ራሴን መጉዳት", "ሕፃኑን መጉዳት", 
    "ራሴን እገድላለሁ", "መኖር ሰልችቶኛል", "ራሴን አጠፋለሁ"
]

def check_safety(user_input):
    return any(kw in user_input.lower() for kw in CRISIS_KEYWORDS)

CRISIS_RESPONSE = """🌸 **You are not alone / እርስዎ ብቻዎን አይደሉም**

I hear your pain. Please get immediate help in Ethiopia:
ከባድ ጭንቀት ውስጥ ከሆኑ እባክዎ ወዲያውኑ እርዳታ ያግኙ፦

- 📞 **የአደጋ ጊዜ ስልክ:** በ **908** ወይም **991** ይደውሉ
- 🏥 **ሆስፒታል:** በአቅራቢያዎ ወደሚገኝ የድንገተኛ ክፍል ይሂዱ (ለምሳሌ፦ አማኑኤል የአእምሮ ስፔሻላይዝድ ሆስፒታል)
- 💕 **ብቻዎን አይሁኑ / Do not stay alone**

ይህ ስሜት በትክክለኛ እርዳታ ያልፋል። 💕"""

# ==========================================
# RAG SETUP (NO STREAMLIT CACHE DECORATOR)
# ==========================================
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = PGVector(
        connection_string=os.getenv("DATABASE_URL"),
        collection_name="mamaspace_docs",
        embedding_function=embeddings
    )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )
    return db, llm

SYSTEM_PROMPT = """You are MamaSpace, a warm, supportive, and deeply empathetic AI clinical self-care companion for Ethiopian mothers navigating postpartum recovery.

CRITICAL TEXT GENERATION RULES:
1. Speak ONLY in natural, fluent, and grammatically perfect Amharic.
2. NEVER repeat or integrate the user's input string directly into your opening validation sentences.
3. Keep sentences very short and direct. Do not try to combine multiple abstract thoughts into long run-on sentences.
4. Structure your clinical self-care response cleanly using short paragraphs and bullet points.

RESPONSE STRUCTURE OUTLINE (Follow this exact framework):
- **የማጽናኛ መልእክት (Validation):** Start with a direct, warm, short validation phrase.
- **ተግባራዊ የጤና ምክሮች (Practical Self-Care Tips):** Provide 3 clear bullet points tailored to managing postpartum recovery independently.
- **መቼ የሕክምና እርዳታ ያስፈልጋል? (Clinical Warning Red Flags):** Provide 1-2 points detailing when she should immediately walk to a Woreda health center."""
