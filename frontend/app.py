import os
from typing import Iterator, Dict, Any, Optional
import time

import streamlit as st
from pypdf import PdfReader

# --- Page config ---
st.set_page_config(page_title="Policy Checker Chatbot", page_icon="✅", layout="wide")

# --- Utilities ---
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

def stream_openai_chat(messages) -> Iterator[str]:
    # messages is the input into LLM
    """Yield content tokens from OpenAI Chat Completions streaming."""

    # insert messages into LLM to get reply. 
    # Insert reply from LLM here as resp.

    # Create a mock response structure that mimics OpenAI's chunk format
    mock_response = type('MockResponse', (), {
        'choices': [type('MockChoice', (), {
            'delta': type('MockDelta', (), {
                'content': "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus non elit sit amet eros fermentum viverra. Morbi at fermentum nisl. In sagittis volutpat aliquet. Sed porta ac mauris et semper. Nam aliquet nec tortor nec scelerisque. Curabitur ultricies mattis sem, vitae accumsan eros feugiat ut. Proin ornare fringilla ante vitae pellentesque. Mauris placerat nisi interdum luctus bibendum. Nulla ut neque ornare ligula molestie sollicitudin. Curabitur commodo massa vitae lectus dignissim, ac suscipit est auctor. Cras ac luctus urna, ac dictum sem. Integer quis libero quam. Nunc varius ipsum orci, vitae malesuada purus volutpat quis. Fusce fringilla ac magna ac volutpat. Nam faucibus elit in condimentum lacinia. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Etiam rhoncus risus sed lorem posuere, eu consequat libero bibendum. Aenean mollis iaculis faucibus. Duis ullamcorper enim in condimentum efficitur. Donec at ligula mauris. Vestibulum et ipsum tellus. Duis in dignissim metus. Ut convallis metus quis purus ultricies hendrerit quis non massa. Maecenas ligula nisi, sagittis nec commodo vel, laoreet non ante. Proin lacinia nulla id tincidunt fermentum. Suspendisse vel libero tortor. Fusce dapibus felis a eros molestie maximus. Donec consequat sodales libero, a commodo magna varius eget. Proin a risus nec massa consequat efficitur eu a ipsum. Praesent arcu est, volutpat sit amet sagittis at, imperdiet ut velit. Maecenas quis libero ac ligula hendrerit consectetur molestie mollis quam. Maecenas nec sagittis ante. Sed non sodales orci. Aenean commodo scelerisque nisi, nec sagittis arcu cursus nec. Cras eget nunc elit."
            })()
        })()]
    })()
    
    # Split the content into chunks to simulate streaming
    content = mock_response.choices[0].delta.content
    words = content.split()
    chunk_size = 10  # words per chunk
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words) + " "
        
        # Simulate realistic streaming delay
        time.sleep(0.03)  
        
        yield chunk_text

def build_violation_prompt(policy_texts: dict, doc_text: str, user_query: str) -> str:
    """
    Compose a single prompt asking the LLM to check violations,
    combining all uploaded policy texts into one corpus.
    """
    # Concatenate all policy texts, labeling each by filename
    combined_policies = ""
    for policy_name, policy_text in policy_texts.items():
        combined_policies += f"\n--- POLICY FILE: {policy_name} ---\n{policy_text[:4000]}\n"
    instruction = f"""
You are a compliance assistant. You are given:
1) POLICY (authoritative rules) - a set of one or more policy documents
2) DOCUMENT (the user's uploaded content)
3) USER PROMPT (what the user is asking now)

TASKS:
- Determine if the DOCUMENT violates the POLICY in any way, or is substantially similar to prohibited content.
- Explain reasoning concisely.
- If there is a violation OR high similarity, set the flags below accordingly.
- ONLY consider the provided POLICY as the source of truth; do not invent rules.

Return your answer starting with a single JSON line (no code fences), with keys:
  is_violation: boolean
  similarity_score: number between 0 and 1 (your confidence that the document matches prohibited patterns)
  rationale: short string (max 60 words)
Then, after the JSON line, continue with a clear, human-friendly explanation.

POLICY CORPUS:{combined_policies}

DOCUMENT:
{doc_text[:8000]}

USER PROMPT:
{user_query}
"""
    return instruction


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta?}
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "policy_texts" not in st.session_state:
    st.session_state.policy_texts = {}
if "decisions" not in st.session_state:
    # map of assistant_message_index -> {is_violation, similarity_score, decision}
    st.session_state.decisions = {}


# --- Sidebar (inputs) ---
with st.sidebar:
    st.header("📄 Input")
    inputs_uploaded = st.file_uploader("Upload a PDF (or use text input below)", type=["pdf"], accept_multiple_files=False)
    text_fallback = st.text_area("…or paste text here", value=st.session_state.doc_text, height=160)

    st.divider()
    st.header("📜 Policy")
    policy_uploaded = st.file_uploader("Upload a PDF(s) of which policies you wish to refer to.", type=["pdf"], accept_multiple_files=True)

    st.divider()
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.decisions = {}
  

# --- Ingest PDFs ---

# Ingest uploaded PDF (to be checked)
if inputs_uploaded is not None:
    with st.spinner("Reading PDF…"):
        pdf_text = extract_text_from_pdf(inputs_uploaded)
        if pdf_text:
            st.session_state.doc_text = pdf_text
            st.toast("PDF loaded ✅", icon="✅")
        else:
            st.session_state.doc_text = ""

# If no PDF, use text fallback
if not st.session_state.doc_text and text_fallback:
    st.session_state.doc_text = text_fallback

# Ingest uploaded PDFs (policies to be used)
policy_texts = {}
if policy_uploaded is not None:
    with st.spinner("Reading PDF(s)…"):
        for uploaded_file in policy_uploaded:
            pdf_name = uploaded_file.name
            pdf_text = extract_text_from_pdf(uploaded_file)
            policy_texts[pdf_name] = pdf_text
    st.session_state.policy_texts = policy_texts

# If no PDFs, use text fallback for a default policy
if not policy_texts and text_fallback:
    st.session_state.policy_texts = {"text_fallback": text_fallback}


# --- Main layout ---
st.title("Policy Checker Chatbot")
st.write("Upload a PDF or paste text, then ask the chatbot to check against your policy. Responses stream in real time.")

# Show a compact info panel
with st.expander("Current Inputs", expanded=False):
    st.markdown("**Policy:**\n" + (str(st.session_state.policy_texts.keys())))
    st.markdown("**Document (first 300 chars):**\n" + (st.session_state.doc_text or text_fallback)[:300])

# Render existing chat history
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If this assistant message had a flag, expose Accept/Decline UI
            meta = msg.get("meta") or {}
            if msg["role"] == "assistant" and meta.get("is_flagged"):
                is_violation = meta.get("is_violation", False)
                similarity = meta.get("similarity", 0.0)
                status = "🚨 Possible VIOLATION" if is_violation else "⚠️ High SIMILARITY"
                st.warning(f"{status} (score: {similarity:.2f}). Please Accept or Decline.")

                left, right = st.columns(2)
                with left:
                    if st.button("✅ Accept", key=f"accept_{i}"):
                        st.session_state.decisions[i] = {
                            "is_violation": is_violation,
                            "similarity_score": similarity,
                            "decision": "accepted",
                        }
                        st.toast("Accepted", icon="✅")
                with right:
                    if st.button("❌ Decline", key=f"decline_{i}"):
                        st.session_state.decisions[i] = {
                            "is_violation": is_violation,
                            "similarity_score": similarity,
                            "decision": "declined",
                        }
                        st.toast("Declined", icon="❌")

# Chat input
user_query = st.chat_input("Ask to check for policy violations, or ask a follow-up…")

if user_query:
    # Persist inputs to state so the assistant uses the freshest copies
    if not st.session_state.policy_texts:
        st.session_state.policy_texts = policy_texts
    if not st.session_state.doc_text:
        st.session_state.doc_text = text_fallback

    st.session_state.messages.append({"role": "user", "content": user_query})

    # System instruction for general helpful behavior (chat context)
    system_msg = {
        "role": "system",
        "content": (
            "You are a concise compliance assistant. Answer clearly. If you are unsure, say so. "
            "Prefer short, actionable guidance. When asked to check for violations, first output a one-line JSON header as instructed, then explain."
        ),
    }

    # Build the violation-check prompt
    composed_prompt = build_violation_prompt(
        policy_texts=st.session_state.policy_texts or policy_texts,
        doc_text=st.session_state.doc_text or text_fallback,
        user_query=user_query,
    )

    # Stream reply
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_text = ""
        for token in stream_openai_chat([
            system_msg,
            {"role": "user", "content": composed_prompt},
        ]):
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
        "content": full_text,
        "meta": {
            "is_flagged": is_flagged,
            "is_violation": is_violation,
            "similarity": similarity,
        },
    })

    # Rerun to show buttons for the last message
    st.rerun()
