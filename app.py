import streamlit as st
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# ==========================================
# PAGE CONFIG & THEME
# ==========================================
st.set_page_config(page_title="MamaSpace 🤱", page_icon="🤱", layout="centered")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #FFF0F5 0%, #FDF5E6 50%, #F0FFF0 100%) !important; }
    [data-testid="stSidebar"] { background: rgba(255, 228, 225, 0.6) !important; }
    .stMarkdown, p, h1, h2, h3 { color: #5D4037 !important; font-family: 'Georgia', serif !important; }
    .stChatMessage { 
        background-color: rgba(255, 255, 255, 0.95) !important; 
        border-radius: 20px !important; 
        border: 1.5px solid #FFD1DC !important;
        color: #5D4037 !important;
    }
    .stChatMessage[user] { 
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%) !important; 
        border: 2px solid #FFB6C1 !important; 
    }
    footer, #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SAFETY GUARDRAILS (ENGLISH + AMHARIC)
# ==========================================
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "hurt myself", "harm myself", 
    "hurt the baby", "harm the baby", "end my life", 
    "postpartum psychosis", "hallucinations", "want to die",
    "ራሴን ማጥፋት", "ራሴን መጉዳት", "ሕፃኑን መጉዳት", "ራሴን እገድላለሁ", "መኖር ሰልችቶኛል", "እራሴን አጠፋለሁ"
]

def check_safety(user_input):
    return any(kw in user_input.lower() for kw in CRISIS_KEYWORDS)

CRISIS_RESPONSE = """🌸 **You are not alone, and your life is precious.** / **እርስዎ ብቻዎን አይደሉም፣ ህይወትዎ እጅግ ውድ ነው።**

I hear how much pain you are in right now. Because you mentioned thoughts of harm, please get immediate support in Ethiopia:
/ እጅግ በጣም እየተቸገሩና እየተሰቃዩ እንደሆነ ይሰማኛል። ራሰዎን ወይም ህፃኑን የመጉዳት ሃሳብ ስላነሱ፣ እባክዎ በኢትዮጵያ ውስጥ በአቅራቢያዎ ከሚገኙ አካላት አስቸኳይ እርዳታ ያግኙ፡-

- 🚑 **Emergency / አስቸኳይ ስልክ:** Call **908** or **991** / **908** ወይም **991** ላይ ይደውሉ
- 🏥 **Hospital / ሆስፒታል:** Go to the nearest emergency room (e.g., Amanuel Mental Specialized Hospital) / በአቅራቢያዎ ወደሚገኝ የድንገተኛ አደጋ ክፍል ይሂዱ (ለምሳሌ፡ አማኑኤል የአእምሮ ስፔሻላይዝድ ሆስፒታል)
- 🤝 **Local Support / የአካባቢ ድጋፍ:** Contact your Woreda health center or a trusted doctor / የአካባቢዎን ጤና ጣቢያ ወይም የሚተማመኑበትን ሀኪም ያማክሩ
- 💕 **Do not stay alone / ብቻዎን አይሁኑ:** Call a family member or friend to sit with you right now / አብሮዎት እንዲሆን በአቅራቢያዎ የሚገኝ የቤተሰብ አባል ወይም ጓደኛ ይጥሩ

You are a wonderful mother. This feeling will pass with help. 💕
/ እርስዎ ድንቅ እናት ነዎት። ይህ አስቸጋሪ ስሜት በትክክለኛ ድጋፍ ያልፋል። 💕"""

# ==========================================
# RAG & GROQ SETUP (CACHED)
# ==========================================
@st.cache_resource
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
        temperature=0.4, # Lowered temperature slightly to keep output more stable and precise
    )
    return db, llm

# Explicitly rewrote the language directives to stop mechanical, broken phrasing.
SYSTEM_PROMPT = """You are MamaSpace, a warm, empathetic AI companion for mothers experiencing postpartum mental health challenges in Ethiopia.

AMHARIC LANGUAGE CONSTRAINTS:
- If the user writes in Amharic, respond ONLY in natural, fluent, and compassionate Amharic.
- Use standard, grammatically correct polite speech (using suffix forms like 'ነዎት' or 'አይደሉም' naturally at the end of sentences).
- NEVER repeat phrases like "እርስዎ ነዎት" or "እርስዎ" at the beginning of clauses. Avoid word-for-word translations from English.
- Speak like a comforting, mature elder sister or an empathetic Ethiopian mother.
- Keep sentences short, comforting, and easily readable.

CONTENT RULES:
1. NEVER give medical advice, diagnose, or prescribe. Direct to doctors for medication.
2. Always validate feelings first with a gentle, non-judgmental tone (e.g., "ይህ ስሜትዎ ፍጹም የተለመደ ነው...").
3. Use provided context from trusted sources. If unsure, say so gently.
4. Use emojis sparingly for warmth (🌸, 💕, 🤱)."""

def mama_chat(message):
    if check_safety(message):
        return CRISIS_RESPONSE
    
    db, llm = load_rag()
    docs = db.similarity_search(message, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    full_prompt = f"""{SYSTEM_PROMPT}

Context from trusted medical sources:
{context}

Mother's message: {message}

Your gentle response:"""
    
    response = llm.invoke(full_prompt)
    return response.content

# ==========================================
# CHAT INTERFACE WITH AMHARIC EXAMPLES
# ==========================================
st.title("MamaSpace 🤱")
st.markdown("<p style='font-size: 18px; color: #5D4037;'>A gentle, safe space for your postpartum journey. / ከወሊድ በኋላ ላለው ጉዞዎ ምቹ እና አስተማማኝ ማረፊያ። 💕</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, mama. 🌸 How are you feeling today? / ሰላም እናት፣ ዛሬ እንዴት ነዎት? ምንስ እየተሰማዎት ነው? 💕"}]

for msg in st.session_state.messages:
    avatar = "💕" if msg["role"] == "assistant" else "👩"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Share what's on your heart... / በልብዎ ያለውን ሃሳብ ያካፍሉኝ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👩"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="💕"):
        with st.spinner("Listening... / እያዳመጥኩዎት ነው... 🌸"):
            response = mama_chat(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
