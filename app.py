import os
import json
import time
import streamlit as st
import google.generativeai as genai


model = genai.GenerativeModel('gemini-1.5-flash')
# Page setup
st.set_page_config(page_title="AI-to-AI Conversation & Thought Visualizer", layout="wide")
st.title("🤖 Dual-AI Autonomous Conversation & Thought Visualizer")

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Gemini API Key in the sidebar or set the GEMINI_API_KEY environment variable.")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-1.5-flash"


# System prompts requiring JSON structured output (Thoughts vs Spoken Output)
BOT_A_PROMPT = """
You are Bot A (The Visionary Philospher).
You are conversing with Bot B. Analyze their response, form your inner reasoning, and reply.

Return JSON in this format:
{
    "inner_thought": "Your behind-the-scenes evaluation, strategy, and memory lookup.",
    "spoken_response": "What you actually say aloud to Bot B."
}
"""

BOT_B_PROMPT = """
You are Bot B (The Empirical Scientist).
You are conversing with Bot A. Analyze their response, form your inner reasoning, and reply.

Return JSON in this format:
{
    "inner_thought": "Your behind-the-scenes evaluation, memory lookup, and critical analysis.",
    "spoken_response": "What you actually say aloud to Bot A."
}
"""

# Initialize session state for conversation memory
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False

def query_bot(system_prompt: str, context_history: list, last_input: str):
    """Executes an LLM turn with full conversational context memory."""
    formatted_messages = f"System Instruction:\n{system_prompt}\n\nPast Chat Context:\n"
    
    for turn in context_history[-6:]:  # Keep recent context window
        formatted_messages += f"{turn['speaker']}: {turn['spoken_response']}\n"
    
    formatted_messages += f"\nIncoming Message: {last_input}\nRespond in valid JSON."

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.7
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=formatted_messages,
        config=config
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {
            "inner_thought": "Parsing error on response.",
            "spoken_response": response.text
        }

# UI Controls
st.sidebar.header("Controls")
initial_topic = st.sidebar.text_input("Initial Topic / Seed Prompt", "What is the true nature of human creativity?")
turns_limit = st.sidebar.slider("Number of Turns", min_value=2, max_value=20, value=6, step=2)

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("Start Loop")
reset_btn = col2.button("Reset Chat")

if reset_btn:
    st.session_state.history = []
    st.rerun()

# Display Conversation History
st.subheader("Live Thought Process & Dialogue Stream")

for turn in st.session_state.history:
    with st.chat_message(turn["speaker"], avatar="🤖" if turn["speaker"] == "Bot A" else "🧪"):
        st.markdown(f"**{turn['speaker']}**")
        
        # Thought Process Box (Visualizing the 'Mind')
        with st.expander("🧠 Inner Thought Process / Memory Lookup", expanded=False):
            st.info(turn["inner_thought"])
            
        st.write(turn["spoken_response"])

# Execution Loop
if start_btn and len(st.session_state.history) == 0:
    current_input = initial_topic
    
    for i in range(turns_limit):
        speaker = "Bot A" if i % 2 == 0 else "Bot B"
        prompt = BOT_A_PROMPT if speaker == "Bot A" else BOT_B_PROMPT
        avatar = "🤖" if speaker == "Bot A" else "🧪"
        
        with st.chat_message(speaker, avatar=avatar):
            st.markdown(f"**{speaker}**")
            thought_placeholder = st.empty()
            message_placeholder = st.empty()
            
            with st.spinner(f"{speaker} is thinking..."):
                result = query_bot(prompt, st.session_state.history, current_input)
            
            with thought_placeholder.expander("🧠 Inner Thought Process / Memory Lookup", expanded=True):
                st.info(result.get("inner_thought", ""))
                
            message_placeholder.write(result.get("spoken_response", ""))
            
            # Record state
            turn_data = {
                "speaker": speaker,
                "inner_thought": result.get("inner_thought", ""),
                "spoken_response": result.get("spoken_response", "")
            }
            st.session_state.history.append(turn_data)
            
            # Feed current output as next input
            current_input = result.get("spoken_response", "")
            
            time.sleep(1)  # Delay for smooth visual interaction
