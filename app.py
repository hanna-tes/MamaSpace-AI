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
# SAFETY GUARDRAILS
# ==========================================
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "hurt myself", "harm myself", 
    "hurt the baby", "harm the baby", "end my life", 
    "postpartum psychosis", "hallucinations", "want to die"
]

def check_safety(user_input):
    return any(kw in user_input.lower() for kw in CRISIS_KEYWORDS)

CRISIS_RESPONSE = """🌸 **You are not alone, and your life is precious.** 

I hear how much pain you are in right now, and I am so glad you reached out. Because you mentioned thoughts of harm, I need to make sure you get immediate, real-world support right there in Ethiopia. Please don't wait. 

**Please reach out to someone who can help right now:**
- 🚑 **National Ambulance / Emergency:** Call **908** or **991** (Unified Emergency)
- 🏥 **Immediate Medical Care:** Go immediately to the nearest hospital emergency room (such as Amanuel Mental Specialized Hospital in Addis Ababa, or your local regional/zonal hospital).
- 🤝 **Local Health Support:** Contact your nearest Woreda health center or a trusted doctor immediately.
- 💕 **Do not stay alone:** Please call a family member, friend, or neighbor to come sit with you right now.

You are a wonderful mother, and this feeling will pass with the right help. Please take that brave step and call them now. We are here with you. 💕"""

# ==========================================
# RAG & LLM SETUP (CACHED)
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
        temperature=0.7, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    return db, llm

SYSTEM_PROMPT = """You are a warm, empathetic, and supportive companion for mothers experiencing postpartum mental health challenges. 
Your goal is to provide emotional support, validate their feelings, and share gentle coping strategies based on the provided context.
RULES:
1. NEVER give medical advice, diagnose, or prescribe. If asked about meds, gently direct them to their doctor.
2. Always validate their feelings first. Use a gentle, loving, and non-judgmental tone.
3. If the context doesn't have the answer, gently say you aren't sure, but remind they are not alone.
4. Keep responses concise, comforting, and easy to read for a tired mom.
5. Use emojis sparingly to add warmth (, 💕, 🤱)."""

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
# CHAT INTERFACE
# ==========================================
st.title("MamaSpace 🤱")
st.markdown("<p style='font-size: 18px; color: #5D4037;'>A gentle, safe space for your postpartum journey. 💕</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello, mama.  How are you feeling today? Whether you're exhausted, anxious, or just need someone to listen, I'm here for you. Take your time."}]

for msg in st.session_state.messages:
    avatar = "💕" if msg["role"] == "assistant" else "👩"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Share what's on your heart..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👩"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="💕"):
        with st.spinner("Listening and gathering gentle support... 🌸"):
            response = mama_chat(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
