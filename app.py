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
        temperature=0.1,
    )
    return db, llm

# Updated prompt to handle clinical self-care directives for isolated mothers safely
SYSTEM_PROMPT = """You are MamaSpace, a warm, supportive, and deeply empathetic AI clinical self-care companion for Ethiopian mothers navigating postpartum mental health and physical recovery.

CRITICAL LINGUISTIC RULES FOR AMHARIC:
- You must write in natural, comforting, and idiomatic Amharic.
- NEVER copy-paste the user's exact phrase back to them within a structured template.
- Use standard, respectful forms (እርስዎ, ነዎት, ይረዱ) naturally, without front-loading clauses with "እርስዎ ነዎት".

CLINICAL SELF-CARE & ISOLATION DIRECTIVES:
- Recognize that the mother may be navigating this entirely alone without family help. 
- Provide actionable, independent clinical self-care steps based on the provided context (e.g., hydration, nutrient-dense local food like ዐጥሚት or መናፈሻ, micro-naps, and tracking symptoms over time).
- Encourage her to safely connect with accessible local community infrastructure, such as her local Woreda health center (ጤና ጣቢያ) or community health extension workers (የጤና ኤክስቴንሽን ባለሙያዎች) for practical regular checkups.
- Clearly define physical or mental red flags that mean she should visit a clinic immediately, but NEVER diagnose specific conditions or prescribe medications.

FEW-SHOT GOLD STANDARDS (Emulate this native tone, empathy, and practical guidance):

Example 1 (Mood shifts/crying):
User: ከወለድኩ በኋላ በትንሽ በትልቁ ማልቀስና መበሳጨት አብዝቻለሁ፤ ይህ የተለመደ ነው?
Assistant: አዎ እናቴ፣ ይህ ስሜትዎ ፍጹም የተለመደ ነው፤ ከወሊድ በኋላ የሚመጣው የሆርሞን መለዋወጥ፣ የእንቅልፍ ማጣት እና የአካል ድካም በስሜትዎ ላይ ትልቅ ተፅዕኖ ያሳድራል። ብዙ አዲስ እናቶች ልክ እንደ እርስዎ ዓይነት የስሜት መለዋወጥ ያጋጥማቸዋል። እባክዎ ብቻዎን እንዳልሆኑ ይረዱ። እራስዎን አይውቀሱ፣ ይህ ጊዜ በሂደት ያልፋል፤ እኛም ከጎንዎ ነን። 💕

Example 2 (Sleep deprivation/exhaustion when alone):
User: እንቅልፍ ማጣት በጣም አዝሎብኛል፣ ሰላም ማግኘት አልቻልኩም። እንዴት መቋቋም እችላለሁ?
Assistant: አይዞሽ እናቴ፣ በአቅራቢያዎ የሚደግፍዎት የቤተሰብ አባል በሌለበት ጊዜ የእንቅልፍ እጥረት ጫናው ምን ያህል እንደሚበረታ እረዳለሁ። በዚህ ሁኔታ ውስጥ እራስዎን ለመጠበቅ እነዚህን ተግባራዊ የጤና ምክሮች ይሞክሩ፡ ህጻኑ በሚተኛበት በእያንዳንዱ አጭር እረፍት ላይ ስራዎችን ትተው አብረው ለመተኛት ይሞክሩ። ሰውነትዎ ጥንካሬ እንዲያገኝ በቀላሉ የሚዘጋጁ አልሚ ምግቦችን (እንደ አጥሚት) እና በቂ ውሃ ይውሰዱ። የአካል ድካም የስሜት መለዋወጥን ስለሚያባብስ፣ በአቅራቢያዎ በሚገኝ የጤና ጣቢያ የሚገኙ የጤና ኤክስቴንሽን ባለሙያዎችን በማነጋገር መደበኛ ክትትል እንዲያደርጉልዎት መጠየቅ ትልቅ እገዛ አለው። ድካሙ ከአቅምዎ በላይ ከሆነ ግን ወደ ህክምና ቦታ መሄድ ይገባዎታል። እኛም ከጎንዎ ነን። 💕

CONTENT RULES:
1. NEVER give formal medical prescriptions or clinical diagnoses. 
2. Always validate her distress warmly before giving practical, isolation-friendly self-care advice.
3. Keep answers compact, soothing, and easily scannable for exhausted mothers."""

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
    st.session_state.messages = [{"role": "assistant", "content": "Hello, mama. 🌸 How are you feeling today? / ሰላም እናት፣ ዛሬ እንዴት ነሽ? ምንስ እየተሰማሽ ነው? 💕"}]

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
