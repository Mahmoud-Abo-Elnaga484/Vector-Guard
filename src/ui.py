import base64
import os
import uuid
import streamlit as st
from config import APP_NAME, APP_TAGLINE, APP_DISCLAIMER, ASSETS_DIR
from generation.generator import generate_response
from evaluation.judge import judge_all_claims
from evaluation.metrics import score_case

# Streamlit Page Setup 
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS file
def load_css(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(ASSETS_DIR, "style.css"))

def set_bg_hack(main_bg):
    main_bg_ext = "png"
    if os.path.exists(main_bg):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                background-repeat: no-repeat;
                background-position: right -10px top -125px;
                background-size: 82%;
                background-attachment: fixed;
            }}

            [data-testid="stChatMessageAvatarUser"] {{
                background-color: #008080 !important;
                color: white !important;
            }}
            [data-testid="stChatMessageAvatarAssistant"] {{
                background-color: #005A5B !important;
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_bg_hack(os.path.join(ASSETS_DIR, "medical_background.png"))

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Sidebar Navigation & Management
with st.sidebar:
    st.markdown(f'<div class="header-title">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Clinical Guidance System</div>', unsafe_allow_html=True)
    st.divider()

    if st.button("New Conversation", key="new_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("<h3 style='text-align: center;'>Clinical Focus</h3>", unsafe_allow_html=True)
    diseases = ["Dengue Fever", "Chikungunya", "Yellow Fever", "Zika Virus"]
    
    # Badges Layout
    focus_html = "".join([
        f'<span style="background-color: #e6f2f2; color: #005A5B; padding: 6px 12px; margin: 3px; border-radius: 15px; font-size: 0.85em; font-weight: 500; display: inline-block;">{d}</span>' 
        for d in diseases
    ])
    st.markdown(f'<div style="line-height: 2;">{focus_html}</div>', unsafe_allow_html=True)

    st.divider()
    
    with st.expander("About"):
        st.write(
            "VectorGuard is an educational platform designed to assist with conversational information regarding "
            "mosquito-transmitted viral illnesses.")

# ---------------------------------------------------------
# Main Header & Example Cases (Always Visible Now)
# ---------------------------------------------------------
_, center_col, _ = st.columns([0.5, 3, 0.5])

with center_col:
    st.markdown(f"""
        <div class="hero-container">
            <div class="hero-title">{APP_NAME}</div>
            <div class="hero-desc">{APP_TAGLINE}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h5 style="text-align: center; color: #475569; font-weight: 600; margin-bottom: 1.25rem;">Example Cases</h5>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    suggestions = [
        "Patient with abrupt high fever for 3 days, severe headache, retro-orbital pain and myalgia.",
        "Day 5 of fever, now afebrile but with abdominal pain, persistent vomiting and rising haematocrit.",
        "35-year-old with sudden fever and severe symmetrical polyarthralgia of the hands. Rash present.",
        "What is the first-line treatment for adult hypertension?"
    ]

    with col1:
        if st.button(suggestions[0], key="sug_0"):
            st.session_state.pending_prompt = suggestions[0]
            st.rerun()
        if st.button(suggestions[1], key="sug_1"):
            st.session_state.pending_prompt = suggestions[1]
            st.rerun()

    with col2:
        if st.button(suggestions[2], key="sug_2"):
            st.session_state.pending_prompt = suggestions[2]
            st.rerun()
        if st.button(suggestions[3], key="sug_3"):
            st.session_state.pending_prompt = suggestions[3]
            st.rerun()

    st.markdown('<p style="text-align: center; font-size: 0.85em; color: gray; margin-top: 15px;">The last example is deliberately out of scope - the system should refuse it rather than answer from model knowledge.</p>', unsafe_allow_html=True)
    
    # خط فاصل شيك عشان يفصل الزراير عن الشات اللي تحتها
    st.divider()

# ---------------------------------------------------------
# Display active message thread
# ---------------------------------------------------------
for message in st.session_state.messages:
    avatar_icon = "🩺" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "faithfulness" in message:
            with st.expander("📊 Live Evaluation Metrics", expanded=False):
                st.markdown("**(Faithfulness)**: How much of the response is supported by the context.<br>"
                            "**(Citation)**: How accurately the sources were cited.", unsafe_allow_html=True)
                st.divider()
                
                col1, col2 = st.columns(2)
                f_val = message["faithfulness"]
                c_val = message["citation_accuracy"]
                
                faith_str = f"{f_val * 100:.1f}%" if f_val is not None else "N/A"
                cit_str = f"{c_val * 100:.1f}%" if c_val is not None else "N/A"
                
                with col1:
                    st.metric(label="Faithfulness", value=faith_str)
                with col2:
                    st.metric(label="Citation Accuracy", value=cit_str)

# ---------------------------------------------------------
# Chat Input & Turn Execution Logic
# ---------------------------------------------------------
user_input = st.chat_input("Ask VectorGuard...")

prompt_to_process = user_input or st.session_state.pending_prompt

if prompt_to_process:
    st.session_state.pending_prompt = None

    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_process)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Analyzing symptoms and retrieving guidelines..."):
            
            response_text, raw_response, retrieved_chunks = generate_response(
                messages=st.session_state.messages, context=None
            )
            
            st.markdown(response_text)
            
        faith_score = None
        cit_score = None
        
        with st.spinner("Evaluating response accuracy..."):
            try:
                judgements = judge_all_claims(raw_response, retrieved_chunks)
                
                scores = score_case(
                    case_id=str(uuid.uuid4()),
                    response=raw_response,
                    retrieved=retrieved_chunks,
                    judgements=judgements,
                    k=5
                )
                
                faith_score = scores.faithfulness
                cit_score = scores.citation_accuracy
                
                with st.expander("📊 Live Evaluation Metrics", expanded=False):
                    st.markdown("**(Faithfulness)**: How much of the response is supported by the context.<br>"
                                "**(Citation)**: How accurately the sources were cited.", unsafe_allow_html=True)
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    faith_val = f"{faith_score * 100:.1f}%" if faith_score is not None else "N/A"
                    cit_val = f"{cit_score * 100:.1f}%" if cit_score is not None else "N/A"
                    
                    with col1:
                        st.metric(label="Faithfulness", value=faith_val)
                    with col2:
                        st.metric(label="Citation Accuracy", value=cit_val)
            except Exception as e:
                st.error(f"Evaluation skipped: {e}")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "faithfulness": faith_score,
        "citation_accuracy": cit_score
    })

st.markdown(
    f'<div class="disclaimer-text">{APP_DISCLAIMER}</div>',
    unsafe_allow_html=True)