import streamlit as st
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import base64
logo_path = r"saimedha_demo_app/privexa_logo.png"

base64_logo = base64.b64encode(open(logo_path, "rb").read()).decode()


st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:15px; margin-bottom:10px;">
        <img src="data:image/png;base64,{base64_logo}" width="65" height="65"> 
        <div>
            <h1 style="margin-bottom:0px;">Pivexa AI</h1>
            <p style="margin-top:-10px; font-size:18px; color:grey;">
                private, secure and on-premise
            </p>
        </div>
    </div>
    <hr style="margin-top:5px;">
    """,
    unsafe_allow_html=True
)
import requests
import os

def download_video_if_not_exists(url, save_path="video.mp4"):
    if not os.path.exists(save_path):
        print("Downloading video... please wait.")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Video downloaded.")
    return save_path

# -------------------------
# Load RAG Data (Chunks + FAISS)
# -------------------------
with open("saimedha_demo_app/sai_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = faiss.read_index("saimedha_demo_app/sai_embeddings.index")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# Groq Client
# -------------------------
client = Groq(api_key=st.secrets["secret"]["api"])

SYSTEM_PROMPT = """
You are a video understanding assistant for ecet exam.
if greatings just respond.
You answer ONLY using the retrieved chunks from the video transcript.
If the answer is not found, say: 'I didn't find this in the video.'
explain indetailed if user want.
"""


# -------------------------
# RAG Function
# -------------------------
def rag_search(query, k=2):
    query_emb = embedding_model.encode([query])
    D, I = index.search(query_emb, k)

    retrieved = [chunks[i] for i in I[0]]
    combined = "\n\n".join(retrieved)

    return combined


# -------------------------
# Groq LLM Generation
# -------------------------
def generate_llm_response(user_query, context):
    full_prompt = f"""
SYSTEM:
{SYSTEM_PROMPT}

CONTEXT FROM VIDEO:
{context}

USER QUESTION:
{user_query}

Answer from the context only.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Fast Groq model
        messages=[{"role": "user", "content": full_prompt}])

    return response.choices[0].message.content


# ---------------------------------------------------------
#                   STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Video RAG Chat", layout="wide")

col1, col2 = st.columns([2, 1])

# -------------------------
# LEFT COLUMN → VIDEO
# -------------------------
hf_video_url = "https://huggingface.co/mahigodike/scratch_detection/resolve/main/saimedha_GUNSHOT_BITS.mp4"

video_path = download_video_if_not_exists(hf_video_url, "saimedha_GUNSHOT_BITS.mp4")

with col1:
    st.header("🎥 Video Player")
    st.video(video_path)


# -------------------------
# RIGHT COLUMN → CHAT
# -------------------------
with col2:
    st.header("💬 Chat with Video")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask something about the video...")

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # RAG Search
        context = rag_search(user_input, k=2)

        # LLM Answer
        bot_reply = generate_llm_response(user_input, context)

        # Save bot message
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        st.rerun()

