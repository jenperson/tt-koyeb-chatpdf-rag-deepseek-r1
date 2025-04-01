# app.py
import os
import tempfile
import time
import streamlit as st
from rag import ChatPDF

st.set_page_config(page_title="RAG with DeepSeek R1 running on Tenstorrent")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def display_messages():
    """Display the chat history using st.chat_message with a custom AI avatar."""
    st.subheader("Chat History")
    ai_avatar = "assets/tt_symbol_purple.png" 
    user_avatar = "👩‍💻" 

    for msg_data in st.session_state["messages"]:
        if len(msg_data) == 2:
            msg, is_user = msg_data
            role = "user" if is_user else "assistant"
            avatar = ai_avatar if role == "assistant" else user_avatar
        else:
            msg, role, avatar = msg_data

        with st.chat_message(role, avatar=avatar):
            st.write(msg)


def process_input(): 
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        # Store and display user message
        user_avatar = "👩‍💻" 
        st.session_state["messages"].append((user_text, "user", user_avatar))
        with st.chat_message("user", avatar=user_avatar):
            st.write(user_text)

        # Generate assistant response
        ai_avatar = "assets/tt_symbol_purple.png"
        print(f"max tokens: {st.session_state["max_tokens"]}")
        with st.chat_message("assistant", avatar=ai_avatar):
            with st.spinner("Thinking..."):
                try:
                    agent_text = st.session_state["assistant"].ask(
                        user_text,
                        k=st.session_state["retrieval_k"],
                        score_threshold=st.session_state["retrieval_threshold"],
                        temperature=st.session_state["temperature"],
                        max_tokens=st.session_state["max_tokens"],
                    )
                except ValueError as e:
                    agent_text = str(e)
                st.write(agent_text)

        # Store assistant response with avatar
        st.session_state["messages"].append((agent_text, "assistant", ai_avatar))

    st.session_state["user_input"] = ""

def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
        )
        os.remove(file_path)


def page():
    """Main app page layout."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()

    st.header("RAG with Local DeepSeek R1")


    st.session_state["ingestion_spinner"] = st.empty()

    # Retrieval settings
    with st.sidebar:
        st.subheader("Upload a Document")
        st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        st.subheader("Settings")
        st.session_state["retrieval_k"] = st.slider(
            "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
        )
        st.session_state["retrieval_threshold"] = st.slider(
            "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )
        st.session_state["temperature"] = st.slider(
            "Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1
        )
        st.session_state["max_tokens"] = st.slider(
            "Max Tokens", min_value=100, max_value=1000, value=500, step=50
        )

    # Display chat messages
    display_messages()

    # Chat input field
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state["user_input"] = user_input
        process_input()


if __name__ == "__main__":
    page()