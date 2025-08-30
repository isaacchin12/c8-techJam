import os
from typing import Iterator, Dict, Any, Optional
import time
import json
import ast

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

def _to_dict_from_string(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            candidate = s[start : end + 1]
            return json.loads(candidate)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    return {"reasoning": s}

def _is_valid_payload(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    if not isinstance(d.get("implications"), str) or not d.get("implications").strip():
        return False
    results = d.get("results")
    if not isinstance(results, list) or not results:
        return False
    for item in results:
        if not isinstance(item, dict):
            continue
        reasoning = item.get("reasoning")
        confidence = item.get("confidence", 0)
        if isinstance(reasoning, str) and len(reasoning.strip()) >= 5:
            try:
                conf_val = float(confidence)
            except Exception:
                conf_val = 0.0
            if conf_val >= 0:
                return True
    return False

def get_ollama_json(prompt) -> Dict[str, Any]:
    expanded_prompt = vdb.expand_abbreviations(prompt, vdb.glossary)
    MAX_RETRIES = 2
    for _ in range(MAX_RETRIES + 1):
        raw = vdb.query_ollama(prompt, expanded_prompt, collection, model="llama3")
        data = _to_dict_from_string(raw)
        if _is_valid_payload(data):
            return data
    # Last resort: return parsed best-effort
    return data

def extract_reasoning(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return str(data)
    if isinstance(data.get("reasoning"), str) and data.get("reasoning").strip():
        return data["reasoning"].strip()
    if isinstance(data.get("results"), list) and data["results"]:
        parts = []
        for item in data["results"]:
            if isinstance(item, dict) and isinstance(item.get("reasoning"), str):
                txt = item["reasoning"].strip()
                if txt:
                    parts.append(txt)
        if parts:
            return "\n\n".join(parts)
    return json.dumps(data)

def stream_chunks(text: str, chunk_size: int = 60, delay: float = 0.005) -> Iterator[str]:
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        time.sleep(delay)

def render_model_output(data: Dict[str, Any]):
    implications = data.get("implications", "").strip() if isinstance(data, dict) else ""
    results = data.get("results", []) if isinstance(data, dict) else []
    if implications:
        st.markdown(f"**Implications:** {implications}")
    if isinstance(results, list) and results:
        for i, item in enumerate(results, 1):
            if not isinstance(item, dict):
                continue
            law = item.get("law", "")
            reasoning = (item.get("reasoning") or "").strip()
            highlight = (item.get("highlight") or "").strip()
            supporting_text = (item.get("supporting_text") or "").strip()
            confidence = item.get("confidence", "")
            st.markdown(f"- **Law:** {law}")
            if reasoning:
                st.markdown(f"  - **Reasoning:** {reasoning}")
            if highlight:
                st.markdown(f"  - **Highlight:** “{highlight}”")
            if supporting_text:
                st.markdown(f"  - **Supporting Text:** “{supporting_text}”")
            if confidence != "":
                try:
                    cval = float(confidence)
                    st.markdown(f"  - **Confidence:** {cval}/10")
                except Exception:
                    st.markdown(f"  - **Confidence:** {confidence}")


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta?}
if "input_pdf_text_name" not in st.session_state:
    st.session_state.input_pdf_text_name = ""
if "input_pdf_text" not in st.session_state:
    st.session_state.input_pdf_text = ""
if "policy_texts" not in st.session_state:
    st.session_state.policy_texts = {}
if "check_prompt_shown_for" not in st.session_state:
    # Track whether the Yes/No check prompt has been shown for a given filename
    st.session_state.check_prompt_shown_for = {}

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
        st.session_state.check_prompt_shown_for = {}


  
# --- Ingest PDFs ---

# Ingest uploaded PDF (to be checked)
if inputs_uploaded is not None:
    if inputs_uploaded.name != st.session_state.input_pdf_text_name:
        with st.spinner("Reading PDF…"):
            input_pdf_text = extract_text_from_pdf(inputs_uploaded)
            if input_pdf_text:
                st.session_state.input_pdf_text_name = inputs_uploaded.name
                st.session_state.input_pdf_text = input_pdf_text
                # Mark that the prompt hasn't been shown for this new file yet
                st.session_state.check_prompt_shown_for[inputs_uploaded.name] = False
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

chat_container = st.container()
# ====================CHAT HISTORY + BUTTON FOR ACCEPTANCE===========================
with chat_container:
    # Render stored chat history so messages persist after reruns
    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "assistant")):
            if "content_json" in msg and isinstance(msg["content_json"], dict):
                render_model_output(msg["content_json"])
            else:
                st.markdown(msg.get("content", ""))

    # Only show the check button once per newly uploaded file
    if st.session_state.input_pdf_text and (inputs_uploaded is not None):
        file_name = st.session_state.input_pdf_text_name
        file_content = st.session_state.input_pdf_text

        prompt_pending = not st.session_state.check_prompt_shown_for.get(file_name, False)

        if prompt_pending:
            with st.chat_message("assistant"):
                st.info(f"Would you like the chatbot to check if **{file_name}** breaks any uploaded rules?")
                col1, col2 = st.columns(2)
                yes_clicked = False
                no_clicked = False
                with col1:
                    if st.button("✅ Yes", key=f"check_rules_yes_{file_name}"):
                        yes_clicked = True
                with col2:
                    if st.button("❌ No", key=f"check_rules_no_{file_name}"):
                        no_clicked = True

            # Handle clicks in the same run, then mark prompt as handled
            if yes_clicked:
                st.session_state.check_prompt_shown_for[file_name] = True

                # Compose the prompt for the LLM
                prompt = (
                    f"Check the following file content against the uploaded policy rules. "
                    f"State if the file breaks any rules, and explain why or why not.\n\n"
                    f"File: {file_name}\n"
                    f"Content:\n{file_content}\n\n"
                )

                # Add user message to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Check if '{file_name}' breaks any uploaded rules."
                })

                # Call the LLM, stream reasoning, then render structured output
                with st.chat_message("assistant"):
                    with st.spinner("Checking with policies..."):
                        data = get_ollama_json(prompt)
                        reasoning_text = extract_reasoning(data)
                        stream_placeholder = st.empty()
                        full_text = ""
                        for token in stream_chunks(reasoning_text):
                            full_text += token
                            stream_placeholder.markdown(full_text)
                        # After streaming, show structured view
                        render_model_output(data)

                # Persist assistant response JSON and refresh UI
                st.session_state.messages.append({
                    "role": "assistant",
                    "content_json": data,
                })
                st.rerun()

            elif no_clicked:
                st.session_state.check_prompt_shown_for[file_name] = True
                st.rerun()

    # ===========================================================

# Chat input
user_query = st.chat_input("Ask to check for policy violations, or ask a follow-up…")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Stream reply and render structured output
    with st.chat_message("assistant"):
        with st.spinner("Checking with policies..."):
            data = get_ollama_json(user_query)
            reasoning_text = extract_reasoning(data)
            stream_placeholder = st.empty()
            full_text = ""
            for token in stream_chunks(reasoning_text):
                full_text += token
                stream_placeholder.markdown(full_text)
            render_model_output(data)

    # Save assistant JSON message so it's rendered on re-draw
    st.session_state.messages.append({
        "role": "assistant",
        "content_json": data
    })

    # Rerun to show buttons for the last message
    st.rerun()
