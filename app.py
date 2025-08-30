import os
from typing import Iterator, Dict, Any, Optional
import time

import streamlit as st
from pypdf import PdfReader

import vector_db_querying as vdb

# spin up database
@st.cache_resource
def get_collection():
    return vdb.set_up_chromadb()

collection = get_collection()

# --- Page config ---
st.set_page_config(page_title="Policy Checker Chatbot", page_icon="✅", layout="wide")

# --- Utilities ---
@st.cache_data
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                # fall back to empty string for troublesome pages
                pages.append("")
        return "\n\n".join(pages).strip()
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

def stream_ollama_chat(prompt) -> Iterator[str]:
    """
    Yield content tokens from Ollama's streaming response via query_ollama.
    `messages` is a list of dicts (OpenAI-style), but we will convert to a prompt string.
    """

    for chunk in vdb.query_ollama(prompt, collection, model="llama3"):
        # Optionally, add a small delay to simulate streaming
        time.sleep(0.005)
        yield chunk


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta?}
if "input_pdf_text_name" not in st.session_state:
    st.session_state.input_pdf_text_name = ""
if "input_pdf_text" not in st.session_state:
    st.session_state.input_pdf_text = ""
if "policy_texts" not in st.session_state:
    st.session_state.policy_texts = {}

for k, v in st.session_state.items():
    print(f"{k}: {v}")
    print("-----------------")
print("=================================")

# --- Sidebar (inputs) ---
with st.sidebar:
    st.header("📄 File to Check:")
    inputs_uploaded = st.file_uploader("Upload a PDF (or use text input below)", type=["pdf"], accept_multiple_files=False)
    
    st.divider()
    st.header("📜 Policy")
    policy_uploaded = st.file_uploader("Upload a PDF(s) of which policies you wish to refer to.", type=["pdf"], accept_multiple_files=True)

    st.divider()
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = []

  
# --- Ingest PDFs ---

# Ingest uploaded PDF (to be checked)
if inputs_uploaded is not None:
    if inputs_uploaded.name != st.session_state.input_pdf_text_name:
        with st.spinner("Reading PDF…"):
            input_pdf_text = extract_text_from_pdf(inputs_uploaded)
            if input_pdf_text:
                st.session_state.input_pdf_text_name = inputs_uploaded.name
                st.session_state.input_pdf_text = input_pdf_text
                st.toast(f"{inputs_uploaded.name} uploaded ✅", icon="✅")
            else:
                st.session_state.input_pdf_text = ""

# Ingest uploaded PDFs (policies to be used)
if policy_uploaded is not None:
    with st.spinner("Reading PDF(s)…"):
        if policy_uploaded:
            for uploaded_file in policy_uploaded:
                policy_pdf_name = uploaded_file.name
                if policy_pdf_name not in list(st.session_state.policy_texts.keys()):
                    policy_pdf_text = extract_text_from_pdf(uploaded_file)
                    st.session_state.policy_texts[policy_pdf_name] = policy_pdf_text
                    st.toast(f"{policy_pdf_name} uploaded ✅", icon="✅")


# --- Main layout ---
st.title("Policy Checker Chatbot")
st.write("Upload a PDF or paste text, then ask the chatbot to check against your policy. Responses stream in real time.")

# If no PDFs, use text fallback for a default policy
if not st.session_state.policy_texts:
    st.warning("⚠️ Please upload a policy to be checked against. ⚠️")
else:
    st.success(f"✅ Using the following documents as reference: {', '.join(st.session_state.policy_texts.keys())}")

# Render existing chat history
chat_container = st.container()
# ====================BUTTON FOR ACCEPTANCE===========================
with chat_container:
    # Only show the check button if a file is uploaded and text is present
    if st.session_state.input_pdf_text and (inputs_uploaded is not None):
        file_name = inputs_uploaded.name
        with st.chat_message("assistant"):
            st.info(f"Would you like the chatbot to check if **{file_name}** breaks any uploaded rules?")
            col1, col2 = st.columns(2)
            check_triggered = False
            with col1:
                if st.button("✅ Yes", key="check_rules_yes"):
                    check_triggered = True
            with col2:
                if st.button("❌ No", key="check_rules_no"):
                    check_triggered = False

        if check_triggered:
            # Compose the prompt for the LLM
            policy_context = "\n\n".join(
                [f"{k}:\n{v}" for k, v in st.session_state.policy_texts.items()]
            ) if st.session_state.policy_texts else "No policy provided."
            user_text = st.session_state.input_pdf_text

            prompt = (
                f"Check the following file content against the uploaded policy rules. "
                f"State if the file breaks any rules, and explain why or why not.\n\n"
                f"File: {file_name}\n"
                f"Content:\n{user_text}\n\n"
            )

            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user",
                "content": f"Check if '{file_name}' breaks any uploaded rules."
            })

            # Call the LLM and stream the response
            with st.chat_message("assistant"):
                with st.spinner("Checking with the chatbot..."):
                    stream_placeholder = st.empty()
                    full_text = ""
                    for token in stream_ollama_chat(prompt):
                        full_text += token
                        stream_placeholder.markdown(full_text)

    # ===========================================================

# Chat input
user_query = st.chat_input("Ask to check for policy violations, or ask a follow-up…")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Stream reply
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_text = ""
        for token in stream_ollama_chat(user_query):
            full_text += token
            stream_placeholder.markdown(full_text)

    # After streaming completes, analyze the first line for JSON signal
    is_flagged = False
    is_violation = False
    similarity = 0.0

    # Try to parse the first line as JSON
    first_line, _, rest = full_text.partition("\n")
    import json
    try:
        data = json.loads(first_line)
        is_violation = bool(data.get("is_violation", False))
        similarity = float(data.get("similarity_score", 0.0))
    except Exception:
        # No JSON header; leave unflagged
        rest = full_text

    # Save assistant message with meta so Accept/Decline buttons render on re-draw
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text
    })

    # Rerun to show buttons for the last message
    st.rerun()
