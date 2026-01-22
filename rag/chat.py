# rag/chat.py
from typing import Dict, List, Tuple
from langchain_ollama import ChatOllama
from .store import search, keyword_rerank

SYS = (
 "You are a strict assistant that ONLY answers from the provided meeting excerpts.\n"
 "If the answer is not clearly present, reply exactly: 'Not discussed in meetings.'\n"
 "At the end, include a 'Citations:' list (meeting title, date, source: transcript/minutes).\n"
)

def answer(query: str, scope: str, meeting_meta: Dict) -> Tuple[str, List[Dict]]:
    hits = search(query, k=8, scope=scope, meeting_id=meeting_meta.get("meeting_id",""))
    hits = keyword_rerank(query, hits)
    if not hits:
        return "Not discussed in meetings.", []

    ctx_blocks, cites = [], []
    for d, _ in hits:
        ctx_blocks.append(d.page_content)
        cites.append({
            "title": d.metadata.get("title",""),
            "date": d.metadata.get("date",""),
            "source": d.metadata.get("source","")
        })

    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    prompt = SYS + "\n\nContext:\n" + "\n---\n".join(ctx_blocks) + f"\n\nQuestion: {query}\nAnswer:"
    rsp = llm.invoke(prompt)
    text = (rsp.content or "").strip()

    if not text or "not discussed" in text.lower():
        return "Not discussed in meetings.", cites

    text += "\n\nCitations:\n" + "\n".join([f"- {c['title']} ({c['date']}) [{c['source']}]" for c in cites])
    return text, cites
