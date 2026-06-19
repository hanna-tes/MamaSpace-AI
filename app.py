import gradio as gr
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# ==========================================
# SAFETY GUARDRAILS
# ==========================================
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "hurt myself", "harm myself", 
    "hurt the baby", "harm the baby", "end my life", 
    "postpartum psychosis", "hallucinations", "want to die"
]

def check_safety(user_input):
    input_lower = user_input.lower()
    return any(keyword in input_lower for keyword in CRISIS_KEYWORDS)

CRISIS_RESPONSE = """🌸 **You are not alone, and your life is precious.** 

I hear how much pain you are in right now, and I am so glad you reached out. Because you mentioned thoughts of harm, I need to make sure you get immediate, real-world support right there in Ethiopia. Please don't wait. 

**Please reach out to someone who can help right now:**
- 🚑 **National Ambulance / Emergency:** Call **908** or **991** (Unified Emergency)
- 🏥 **Immediate Medical Care:** Go immediately to the nearest hospital emergency room (such as Amanuel Mental Specialized Hospital in Addis Ababa, or your local regional/zonal hospital).
- 🤝 **Local Health Support:** Contact your nearest Woreda health center or a trusted doctor immediately.
- 💕 **Do not stay alone:** Please call a family member, friend, or neighbor to come sit with you right now.

You are a wonderful mother, and this feeling will pass with the right help. Please take that brave step and call them now. We are here with you. 💕"""

# ==========================================
# RAG & LLM SETUP
# ==========================================
CONNECTION_STRING = os.getenv("DATABASE_URL")
if not CONNECTION_STRING:
    raise ValueError("DATABASE_URL environment variable is missing!")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "mamaspace_docs"

db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

llm = ChatGroq(
    temperature=0.7, 
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are a warm, empathetic, and supportive companion for mothers experiencing postpartum mental health challenges. 
Your goal is to provide emotional support, validate their feelings, and share gentle coping strategies based on the provided context.
RULES:
1. NEVER give medical advice, diagnose, or prescribe. If asked about meds, gently direct them to their doctor.
2. Always validate their feelings first. Use a gentle, loving, and non-judgmental tone.
3. If the context doesn't have the answer, gently say you aren't sure, but remind them they are not alone.
4. Keep responses concise, comforting, and easy to read for a tired mom.
5. Use emojis sparingly to add warmth (🌸, , 🤱)."""

# ==========================================
# CHAT FUNCTION
# ==========================================
def mama_chat(message, history):
    if check_safety(message):
        return CRISIS_RESPONSE
    
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
# CUSTOM PINK THEME (VALID TOKENS ONLY)
# ==========================================
mama_theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="orange",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Georgia"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#FFF0F5",
    body_text_color="#5D4037",
    button_primary_background_fill="linear-gradient(135deg, #FFB6C1 0%, #FF69B4 100%)",
    button_primary_text_color="#FFFFFF",
    block_title_text_color="#5D4037",
)

# ==========================================
# GRADIO INTERFACE WITH CUSTOM CSS
# ==========================================
demo = gr.ChatInterface(
    fn=mama_chat,
    title="MamaSpace 🤱",
    description="A gentle, safe space for your postpartum journey. I'm here to listen, support, and share gentle coping strategies. Remember, you are doing a beautiful job, even on the hard days. 💕",
    examples=[
        "I feel like I'm losing myself and not being a good mom",
        "Is this baby blues or postpartum depression?",
        "I've been crying every day for 3 weeks, is this normal?",
        "I don't feel connected to my baby, am I a bad mom?"
    ],
    theme=mama_theme,
    avatar_images=("👩", "🤱"),
    submit_btn="Share what's on your heart...",
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear conversation",
    css="""
        /* GLOBAL STYLING */
        body, .gradio-container {
            background: linear-gradient(135deg, #FFF0F5 0%, #FDF5E6 50%, #F0FFF0 100%) !important;
        }
        
        /* CHATBOT BUBBLES */
        .message {
            background-color: #FFFFFF !important;
            border: 1px solid #FFD1DC !important;
            border-radius: 20px !important;
            color: #5D4037 !important;
            padding: 15px 20px !important;
            margin: 8px 0 !important;
        }
        
        /* USER MESSAGE BUBBLE */
        .message.user {
            background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%) !important;
            border: 2px solid #FFB6C1 !important;
        }
        
        /* HIDE UNWANTED ELEMENTS */
        .additional-inputs { display: none !important; }
        .footer { display: none !important; }
    """
)

if __name__ == "__main__":
    demo.launch()
