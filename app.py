# ==========================================
# GRADIO INTERFACE WITH CUSTOM THEME
# ==========================================

# Create a custom soft pink theme matching your Streamlit design
mama_theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="orange",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Georgia"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#FFF0F5",
    body_text_color="#5D4037",
    chatbot_color="#FFFFFF",
    button_primary_background_fill="linear-gradient(135deg, #FFB6C1 0%, #FF69B4 100%)",
    button_primary_text_color="#FFFFFF",
    block_title_text_color="#5D4037",
)

demo = gr.ChatInterface(
    fn=mama_chat,
    title="MamaSpace 🤱🌸",
    description="A gentle, safe space for your postpartum journey. I'm here to listen, support, and share gentle coping strategies. Remember, you are doing a beautiful job, even on the hard days. 💕",
    examples=[
        "I feel like I'm losing myself and not being a good mom",
        "Is this baby blues or postpartum depression?",
        "I've been crying every day for 3 weeks, is this normal?",
        "I don't feel connected to my baby, am I a bad mom?"
    ],
    theme=mama_theme,  # ✅ Apply our custom pink theme
    avatar_images=("", "🤱"),  # ✅ Emojis work perfectly in Gradio!
    submit_btn="Share what's on your heart...",
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear conversation",
    css="""
        /* Force light background everywhere */
        body, .gradio-container {
            background: linear-gradient(135deg, #FFF0F5 0%, #FDF5E6 50%, #F0FFF0 100%) !important;
        }
        /* Make chat bubbles white with pink border */
        .message {
            background-color: #FFFFFF !important;
            border: 1px solid #FFD1DC !important;
            border-radius: 20px !important;
            color: #5D4037 !important;
        }
        /* User message bubble - soft pink gradient */
        .message.user {
            background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%) !important;
            border: 2px solid #FFB6C1 !important;
        }
        /* Hide 'Additional inputs' section */
        .additional-inputs { display: none !important; }
    """
)

if __name__ == "__main__":
    demo.launch()
