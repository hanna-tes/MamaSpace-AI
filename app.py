import streamlit as st
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. UI & THEME SETUP ("Mother-Baby Love")
# ==========================================
st.set_page_config(page_title="MamaSpace ", page_icon="🤱", layout="centered")

# Custom CSS for a warm, soft, and comforting aesthetic
st.markdown("""
<style>
    /* Main background: Soft, warm, comforting gradient */
    .stApp {
        background: linear-gradient(135deg, #FFF0F5 0%, #FDF5E6 50%, #F0FFF0 100%) !important;
        min-height: 100vh;
    }
    
    /* Remove ALL dark backgrounds */
    [data-testid="stBottomBlockContainer"], 
    .stChatInputContainer,
    .stChatMessage,
    [data-testid="stBottomContainer"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 228, 225, 0.6) !important;
        backdrop-filter: blur(10px);
    }
    
    /* All text - dark brown for readability */
    .stMarkdown, p, h1, h2, h3, h4, h5, h6, 
    div[data-testid="stMarkdownContainer"] p,
    .stChatMessage .stMarkdown {
        color: #5D4037 !important;
        font-family: 'Georgia', serif !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #5D4037 !important; 
        font-family: 'Georgia', serif !important;
    }
    
    /* Chat input container - WHITE with soft pink border */
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 25px !important;
        padding: 20px 30px !important;
        margin: 20px auto !important;
        max-width: 900px !important;
        border: 2px solid #FFB6C1 !important;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.25) !important;
    }
    
    /* Chat input textarea */
    .stChatInput textarea {
        background-color: #FFFFFF !important;
        border: none !important;
        border-radius: 25px !important;
        color: #5D4037 !important;
        font-size: 16px !important;
        padding: 15px 20px !important;
        box-shadow: none !important;
    }
    
    /* Placeholder text */
    .stChatInput textarea::placeholder {
        color: #A89F91 !important;
        opacity: 0.7 !important;
    }
    
    /* Send button */
    .stChatInput button[kind="secondary"] {
        background: linear-gradient(135deg, #FFB6C1 0%, #FF69B4 100%) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 48px !important;
        height: 48px !important;
        box-shadow: 0 2px 8px rgba(255, 105, 180, 0.3) !important;
    }
    
    /* Chat message bubbles */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        padding: 20px 25px !important;
        margin: 15px auto !important;
        max-width: 850px !important;
        border: 1.5px solid #FFD1DC !important;
        box-shadow: 0 3px 10px rgba(255, 182, 193, 0.15) !important;
    }
    
    /* User message - soft pink gradient */
    .stChatMessage[user] {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%) !important;
        border: 2px solid #FFB6C1 !important;
    }
    
    /* Assistant message - white */
    .stChatMessage:not([user]) {
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Center alignment */
    .element-container, .stChatMessage {
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Hide footer and menu */
    footer, #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid #FFD1DC !important;
        margin: 30px auto !important;
        max-width: 850px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SAFETY GUARDRAILS (Ethiopia Context)
# ==========================================
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "hurt myself", "harm myself", 
    "hurt the baby", "harm the baby", "end my life", 
    "postpartum psychosis", "hallucinations", "want to die"
]

def check_safety(user_input):
    input_lower = user_input.lower()
    return any(keyword in input_lower for keyword in CRISIS_KEYWORDS)

CRISIS_RESPONSE = """
 **You are not alone, and your life is precious.** 🌸

I hear how much pain you are in right now, and I am so glad you reached out. Because you mentioned thoughts of harm, I need to make sure you get immediate, real-world support right there in Ethiopia. Please don't wait. 

**Please reach out to someone who can help right now:**
- 🚑 **National Ambulance / Emergency:** Call **908** or **991** (Unified Emergency)
- 🏥 **Immediate Medical Care:** Go immediately to the nearest hospital emergency room (such as Amanuel Mental Specialized Hospital in Addis Ababa, or your local regional/zonal hospital).
- 🤝 **Local Health Support:** Contact your nearest Woreda health center or a trusted doctor immediately.
- 💕 **Do not stay alone:** Please call a family member, friend, or neighbor to come sit with you right now.

You are a wonderful mother, and this feeling will pass with the right help. Please take that brave step and call them now. We are here with you. 💕
"""

# ==========================================
# 3. RAG & LLM SETUP
# ==========================================
@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    CONNECTION_STRING = "postgresql+psycopg2://localhost/mamaspace_db"
    COLLECTION_NAME = "mamaspace_docs"
    
    # Connect to Postgres Vector DB
    db = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    # Using Groq for fast, free LLM inference
    llm = ChatGroq(
        temperature=0.7, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    return db, llm

# ==========================================
# 4. STREAMLIT CHAT INTERFACE
# ==========================================

# Beautiful header with emojis
st.title("MamaSpace 🤱")
st.markdown("<p style='font-size: 20px; color: #5D4037; font-style: italic;'>A gentle, safe space for your postpartum journey.</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #8B7355; margin-bottom: 30px;'>I'm here to listen, support, and share gentle coping strategies. Remember, you are doing a beautiful job, even on the hard days. 💕</div>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #FFD1DC; margin: 30px 0;'>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, mama. 🌸 How are you feeling today? Whether you're exhausted, anxious, or just need someone to listen, I'm here for you. Take your time."}
    ]

# Display chat messages from history using SAFE emojis
for message in st.session_state.messages:
    # 💕 is safe for assistant, 👩 is safe for user
    avatar = "💕" if message["role"] == "assistant" else "👩"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input with better styling
if prompt := st.chat_input("Share what's on your heart..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👩"):
        st.markdown(prompt)
        
    # --- SAFETY CHECK ---
    if check_safety(prompt):
        response = CRISIS_RESPONSE
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="💕"):
            st.markdown(response)
    else:
        # --- GENERATE RAG RESPONSE ---
        with st.chat_message("assistant", avatar="💕"):
            with st.spinner("Listening and gathering gentle support... 🌸"):
                db, llm = load_rag()
                
                # System prompt enforcing behavioral science & empathy
                system_prompt = """You are a warm, empathetic, and supportive companion for mothers experiencing postpartum mental health challenges. 
                Your goal is to provide emotional support, validate their feelings, and share gentle coping strategies based on the provided context.
                RULES:
                1. NEVER give medical advice, diagnose, or prescribe. If asked about meds, gently direct them to their doctor.
                2. Always validate their feelings first. Use a gentle, loving, and non-judgmental tone.
                3. If the context doesn't have the answer, gently say you aren't sure, but remind them they are not alone.
                4. Keep responses concise, comforting, and easy to read for a tired mom.
                5. Use emojis sparingly to add warmth (🌸, 💕, 🤱)."""
                
                # Retrieve relevant context from Postgres
                docs = db.similarity_search(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                full_prompt = f"""{system_prompt}
                
Context from trusted medical sources:
{context}

Mother's message: {prompt}

Your gentle response:"""
                
                response = llm.invoke(full_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
