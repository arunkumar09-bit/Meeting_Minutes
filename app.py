# app.py
import os
from datetime import datetime
import streamlit as st
from asr.transcribe import transcribe_audio

st.set_page_config(page_title="Minutes Generator (Audio → Minutes)", layout="wide")
st.title("Minutes Generator (Audio → LoRA Minutes)")

# --- Lazy import & cache the summarizer module and model ---
@st.cache_resource(show_spinner=False)
def load_minutes_module_and_model():
    """
    Import summarizer.generate_minutes and load its model/tokenizer.
    Returns (gm_module, model, tokenizer).
    """
    import importlib
    gm = importlib.import_module("summarizer.generate_minutes")

    # must be zero-arg per your implementation
    model, tokenizer = gm.load_model_and_tokenizer()
    # let tokenizer allow long texts (gm handles chunking anyway)
    try:
        tokenizer.model_max_length = 10**9
    except Exception:
        pass
    return gm, model, tokenizer

def run_minutes_pipeline(transcript_text: str, date_str: str):
    """
    Uses your module’s exact API:
    clean_transcript -> summarize_chunks -> synthesize_minutes
    """
    gm, model, tokenizer = load_minutes_module_and_model()

    cleaned = gm.clean_transcript(transcript_text)
    section_summaries = gm.summarize_chunks(model, tokenizer, cleaned)
    minutes = gm.synthesize_minutes(model, tokenizer, section_summaries, date_str or None)

    # If the template literal is still present, replace it
    if "Date: <dd/mm/yyyy>" in minutes and date_str:
        minutes = minutes.replace("Date: <dd/mm/yyyy>", f"Date: {date_str}")

    return cleaned, section_summaries, minutes

def persist_run(meeting_id: str, transcript_text: str, minutes_text: str, date_str: str, title: str):
    out_dir = os.path.join("storage", "meetings", meeting_id)
    os.makedirs(out_dir, exist_ok=True)
    tpath = os.path.join(out_dir, "transcript.txt")
    mpath = os.path.join(out_dir, "minutes.md")
    meta  = os.path.join(out_dir, "meta.txt")
    with open(tpath, "w", encoding="utf-8") as f: f.write(transcript_text)
    with open(mpath, "w", encoding="utf-8") as f: f.write(minutes_text)
    with open(meta, "w", encoding="utf-8") as f:
        f.write(f"title: {title}\ndate: {date_str}\nmeeting_id: {meeting_id}\n")
    return tpath, mpath

# ---- UI ----
col1, col2 = st.columns([2, 1])
with col1:
    date_str = st.text_input("Date (dd/mm/yyyy)", value=datetime.now().strftime("%d/%m/%Y"))
    title = st.text_input("Meeting title", value="Untitled Meeting")
with col2:
    show_notes = st.checkbox("Show per-chunk notes (debug)", value=False)

c1, c2 = st.columns(2)
with c1:
    audio_file = st.file_uploader("Upload meeting audio (.mp3/.wav/.m4a)", type=["mp3", "wav", "m4a"])
    btn_audio = st.button("Transcribe audio ➜ Generate Minutes", use_container_width=True)
with c2:
    pasted_transcript = st.text_area("…or paste transcript text directly", height=220, placeholder="Paste transcript here…")
    btn_text = st.button("Generate Minutes from pasted transcript", use_container_width=True)

# ---- Actions ----
if btn_audio:
    if not audio_file:
        st.error("Please upload an audio file.")
    else:
        tmp_dir = os.path.join("storage", "meetings")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"tmp_{audio_file.name}")
        with open(tmp_path, "wb") as f:
            f.write(audio_file.read())

        with st.spinner("Transcribing…"):
            try:
                transcript_text, _ = transcribe_audio(tmp_path)  # your asr/transcribe.py returns (text, path) in your earlier version
            except TypeError:
                # if your transcriber returns only text
                transcript_text = transcribe_audio(tmp_path)
            finally:
                try: os.remove(tmp_path)
                except Exception: pass

        st.success("Transcription complete.")
        with st.expander("Show transcript"):
            st.text_area("Transcript (raw)", value=transcript_text, height=220)

        with st.spinner("Generating minutes…"):
            cleaned, notes, minutes = run_minutes_pipeline(transcript_text, date_str)

        if show_notes:
            st.subheader("Per-chunk Notes (map step)")
            st.code("\n\n".join(notes), language="markdown")

        st.subheader("Minutes (generated)")
        st.code(minutes, language="markdown")

        meeting_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        tpath, mpath = persist_run(meeting_id, transcript_text, minutes, date_str, title)
        st.session_state["meeting_meta"] = {"meeting_id": meeting_id, "title": title, "date": date_str}

        st.download_button(
            "Download Minutes (.md)",
            data=minutes.encode("utf-8"),
            file_name=f"minutes_{meeting_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.info(f"Saved to:\n- {tpath}\n- {mpath}")

if btn_text:
    if not pasted_transcript.strip():
        st.error("Paste a transcript or upload audio.")
    else:
        with st.spinner("Generating minutes…"):
            cleaned, notes, minutes = run_minutes_pipeline(pasted_transcript, date_str)

        if show_notes:
            st.subheader("Per-chunk Notes (map step)")
            st.code("\n\n".join(notes), language="markdown")

        st.subheader("Minutes (generated)")
        st.code(minutes, language="markdown")

        meeting_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        tpath, mpath = persist_run(meeting_id, pasted_transcript, minutes, date_str, title)
        st.session_state["meeting_meta"] = {"meeting_id": meeting_id, "title": title, "date": date_str}

        st.download_button(
            "Download Minutes (.md)",
            data=minutes.encode("utf-8"),
            file_name=f"minutes_{meeting_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.info(f"Saved to:\n- {tpath}\n- {mpath}")

st.caption("Tip: after updating LoRA weights, click the ‘Rerun’ button or use ‘Clear cache’ to reload the model.")
import streamlit as st
from rag.store import search
from langchain_ollama import ChatOllama

st.header("Ask questions about the meetings")

scope = st.radio("Scope", ["This meeting", "All meetings"], horizontal=True)
q = st.text_input("Your question", placeholder="What was decided about hot water issues?")

# Keep latest meeting meta in session when you generate minutes:
# st.session_state["meeting_meta"] = {"meeting_id": meeting_id, "title": title, "date": date_str}

if st.button("Answer from minutes/transcripts"):
    if not q.strip():
        st.error("Please enter a question.")
    else:
        meeting_meta = st.session_state.get("meeting_meta", {})
        use_scope = "this" if scope == "This meeting" else "all"
        hits = search(q, k=6, scope=use_scope, meeting_id=meeting_meta.get("meeting_id",""))

        if not hits:
            st.warning("Not discussed in meetings.")
        else:
            # Build context
            ctx_blocks = []
            cites = []
            for doc, score in hits[:4]:
                ctx_blocks.append(doc.page_content)
                cites.append(f"- {meeting_meta.get('title','')} ({meeting_meta.get('date','')}) [{doc.metadata.get('source','')}]")

            llm = ChatOllama(model="llama3.1:8b", temperature=0)
            prompt = (
                "Answer strictly from the context. If unsure, say 'Not discussed in meetings.'\n\n"
                "Context:\n" + "\n---\n".join(ctx_blocks) +
                f"\n\nQuestion: {q}\nAnswer:"
            )
            resp = llm.invoke(prompt).content.strip() or "Not discussed in meetings."
            st.write(resp)
            st.caption("Citations:\n" + "\n".join(cites))
