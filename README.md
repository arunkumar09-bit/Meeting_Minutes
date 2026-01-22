# Meeting Minutes & Q&A Agent

**Author:** Pochimireddy Ajith Reddy  
**University:** Indian Institute of Technology Hyderabad  
**Department:** Civil Engineering  

---

## Project Overview


The system automates the generation of **structured Minutes of Meeting (MoM)** from raw meeting transcripts (or audio, via Whisper), and supports **retrieval-augmented Q&A** across past meetings.

**Core capabilities:**
1. **Automatic Minutes Generation**  
   - Converts transcripts into a structured Gymkhana-style format (Issue → Discussion → Decision → Action).  
   - Fine-tuned using **LoRA adapters** on `flan-t5-base` with IIT Hyderabad Gymkhana MoM data.

2. **Meeting Storage & Indexing**  
   - Saves MoMs in `/storage/meetings/` with transcripts + metadata.  
   - Embeds and indexes using **SentenceTransformers + FAISS**.

3. **Q&A over Meetings (RAG)**  
   - Allows asking questions about **current meeting** or **all past meetings**.  
   - Retrieves relevant chunks and composes concise answers with citations.



