import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Missing GROQ_API_KEY in .env file!")
    st.stop()

st.set_page_config(page_title="Muhammad Ali -Chatbot", page_icon="🤖", layout="centered")

# Custom UI
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .main {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }
        .stTextArea textarea {
            color: white !important;
            background-color: rgba(255,255,255,0.1) !important;
        }
        .stButton>button {
            background-color: #00c9ff;
            color: white;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🤖 Muhammad Ali — Personal Chatbot")

# Sidebar settings
st.sidebar.title("⚙️ Settings")
model = st.sidebar.selectbox("Choose a model:", [
    "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
])
memory_length = st.sidebar.slider("Memory Length:", 1, 10, value=5)

# Session state initialization
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "current_session" not in st.session_state:
    st.session_state.current_session = []
if "selected_chat_index" not in st.session_state:
    st.session_state.selected_chat_index = None

# Load selected past chat into main area
if st.session_state.selected_chat_index is not None:
    session_to_view = st.session_state.chat_sessions[st.session_state.selected_chat_index]
    st.subheader(f"📁 Viewing Chat {st.session_state.selected_chat_index + 1}")
    for pair in session_to_view:
        st.markdown(f"**🧑 You:** {pair['human']}")
        st.markdown(f"**🤖 Bot:** {pair['AI']}")
    if st.button("⬅️ Back to New Chat"):
        st.session_state.selected_chat_index = None
        st.rerun()

else:
    # Sidebar chat titles only
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗂 Chats")
    for i, session in enumerate(st.session_state.chat_sessions):
        if st.sidebar.button(f"📄 Chat {i+1}"):
            st.session_state.selected_chat_index = i
            st.rerun()

    # Create memory for new/current chat
    memory = ConversationBufferWindowMemory(k=memory_length)
    for pair in st.session_state.current_session:
        memory.save_context({"input": pair["human"]}, {"output": pair["AI"]})

    # Load LLM
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("💬 Ask something:", height=100)
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        response = conversation(user_input)
        reply = response["response"]
        st.session_state.current_session.append({"human": user_input, "AI": reply})

    # Display current chat (only in main, not sidebar)
    if st.session_state.current_session:
        st.subheader("💬 Current Chat")
        for pair in st.session_state.current_session:
            st.markdown(f"**🧑 You:** {pair['human']}")
            st.markdown(f"**🤖 Bot:** {pair['AI']}")

    # Save session
    if st.button("💾 Save Chat"):
        if st.session_state.current_session:
            st.session_state.chat_sessions.append(st.session_state.current_session.copy())
            st.session_state.current_session = []
            st.success("✅ Chat saved!")
            st.rerun()

# Sidebar clear all button
if st.sidebar.button("🗑️ Clear All Chats"):
    st.session_state.chat_sessions = []
    st.session_state.current_session = []
    st.session_state.selected_chat_index = None
    st.sidebar.success("Chats cleared.")
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
