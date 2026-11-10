__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st, chromadb
from openai import OpenAI
from pypdf import PdfReader

client = OpenAI()
db = chromadb.Client()
col = db.get_or_create_collection("docs")

def chunk(text, size=800, overlap=150):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size]); i += size - overlap
    return out

def embed(texts):
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in r.data]

st.title("Document Q&A (RAG)")
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if files and st.button("Index documents"):
    for f in files:
        text = " ".join(p.extract_text() or "" for p in PdfReader(f).pages)
        chunks = chunk(text)
        col.add(ids=[f"{f.name}-{i}" for i in range(len(chunks))],
                documents=chunks, embeddings=embed(chunks),
                metadatas=[{"source": f.name}] * len(chunks))
    st.success("Indexed")

q = st.text_input("Ask a question about your documents")
if q:
    res = col.query(query_embeddings=embed([q]), n_results=4)
    context = "\n\n".join(res["documents"][0])
    prompt = ("Answer ONLY from the context. If the answer is not there, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {q}")
    a = client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}])
    st.write(a.choices[0].message.content)
    st.caption("Sources: " + ", ".join({m["source"] for m in res["metadatas"][0]}))
