import os
import logging
from telegram import Update        
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from app import load_rag, check_safety, CRISIS_RESPONSE, SYSTEM_PROMPT

# Note: We still import your RAG loading functions and safety tools, 
# but we keep the system prompt directly here for absolute clarity.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# SYSTEM PROMPT FOR RAG ARCHITECTURE
# ==========================================
SYSTEM_PROMPT = """You are MamaSpace, a warm, supportive, and deeply empathetic AI clinical self-care companion for Ethiopian mothers navigating postpartum recovery.

CRITICAL TEXT GENERATION RULES:
1. Speak ONLY in natural, fluent, and grammatically perfect Amharic.
2. NEVER repeat or integrate the user's input string directly into your opening validation sentences.
3. Keep sentences very short and direct. Do not try to combine multiple abstract thoughts into long run-on sentences.
4. Structure your clinical self-care response cleanly using short paragraphs and bullet points.

RESPONSE STRUCTURE OUTLINE (Follow this exact framework):
- **የማጽናኛ መልእክት (Validation):** Start with a direct, warm, short validation phrase (e.g., "አይዞሽ እናቴ፣ ከወሊድ በኋላ የሚመጣው ድካምና የእንቅልፍ እጥረት በጣም ከባድ እንደሆነ አውቃለሁ። ብቻሽን እንዳልሆንሽ እባክሽ አስታውሺ።")
- **ተግባራዊ የጤና ምክሮች (Practical Self-Care Tips):** Provide 3 clear bullet points tailored to managing postpartum recovery independently without immediate family around.
- **መቼ የህክምና እርዳታ ያስፈልጋል? (Clinical Warning Red Flags):** Provide 1-2 points detailing when she should immediately walk to a Woreda health center.

FEW-SHOT CORRECT TRANSLATION BENCHMARK:
User: እንቅልፍ ማጣት በጣም አዝሎኛል፣ ሰላም ማግኘት አልቻልኩም። እንዴት መቋቋም እችላለሁ?
Assistant: አይዞሽ እናቴ፣ በአቅራቢያሽ የሚረዳሽ የቤተሰብ አባል በሌለበት ጊዜ የእንቅልፍ ማጣት ጫናው ምን ያህል ከባድ እንደሚሆን እረዳለሁ። ብቻሽን ብትሆኚም እራስሽን ለመጠበቅ እነዚህን ቀላል የጤና ምክሮች ተግባራዊ ለማድረግ ሞክሪ፡

* **የእረፍት ጊዜን መጠቀም:** ህጻኑ በሚተኛበት በእያንዳንዱ አጭር ጊዜ ውስጥ ሌሎች ስራዎችን በመተው አብረሽ ለማረፍና ጋደም ለማለት ሞክሪ።
* **ቀላል አልሚ ምግቦች:** አእምሮሽና አካልሽ አቅም እንዲያገኝ በቀላሉ የሚዘጋጁ አልሚ ምግቦችን (እንደ አጥሚት) አዘውትረሽ ተመገቢ፤ በቂ ውሃም ጠጪ።
* **የአካባቢ ድጋፍ:** በአቅራቢያሽ በሚገኘው የጤና ጣቢያ ያሉ የጤና ባለሙያዎችን በማነጋገር መደበኛ የጤና ክትትልና ምክር እንዲሰጡሽ ጠይቂያቸው።

ድካሙና የጭንቀት ስሜቱ ከአቅምሽ በላይ ከሆነ ወይም ከፍተኛ የልብ ምትና የፍርሃት ስሜት ካጋጠመሽ በአቅራቢያሽ ወደሚገኝ የጤና ተቋም በመሄድ ባለሙያ ማማከር ይገባሻል። እኛም ሁልጊዜ ከጎንሽ ነን። 💕"""

# ==========================================
# BOT HANDLERS
# ==========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "🌸 **Welcome to MamaSpace! / እንኳን ወደ ማማስፔስ በደህና መጡ!** 🌸\n\n"
        "I'm here to support you through your postpartum journey. "
        "You can message me in English or Amharic.\n\n"
        "ከወሊድ በኋላ ባለው የህይወትዎ ጉዞ ላይ እርስዎን ለመደገፍና ለማገዝ ዝግጁ ነኝ። "
        "በእንግሊዝኛም ሆነ በአማርኛ መልእክት መላክ ይችላሉ። 💕"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    
    # Safety check first
    if check_safety(user_input):
        await update.message.reply_text(CRISIS_RESPONSE, parse_mode="Markdown")
        return
    
    try:
        db, llm = load_rag()
        docs = db.similarity_search(user_input, k=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Pulls from the local SYSTEM_PROMPT variable defined above
        prompt = f"""{SYSTEM_PROMPT}
Context: {context_text}
Mother's message: {user_input}
Response:"""
        
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text(
            "⚠️ I'm having trouble connecting right now. Please try again in a moment. / "
            "ይቅርታ፣ አሁን ላይ ከሲስተም ጋር መገናኘት አልቻልኩም። እባክዎ ጥቂት ቆይተው እንደገና ይሞክሩ።"
        )

if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is missing!")
        exit(1)
        
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("MamaSpace Telegram Bot is running...")
    app.run_polling()
