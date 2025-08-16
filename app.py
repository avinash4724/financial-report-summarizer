# app.py
import os, re, io, tempfile, json
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ----------------------------
# Streamlit App Setup
# ----------------------------
st.set_page_config(page_title="📊 Financial Filings Explorer", layout="wide")
st.title("📊 AI-Powered Financial Filings Explorer")

with st.sidebar:
    st.subheader("🔐 Credentials & Settings")
    groq_api = st.text_input("Groq API Key", type="password")
    
    # Sidebar Model Selection
model_choice = st.sidebar.selectbox(
    "Select Groq Model",
    [
        "llama-3.3-70b-versatile",   # recommended default
        "llama-3.1-8b-instant",
        "mistral-7b",
        "gemma-7b"
    ],
    index=0
)
chunk_size = st.number_input("Chunk size", 200, 4000, 1000, step=100)
chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 200, step=50)
top_k = st.number_input("Retriever k", 1, 10, 3, step=1)
reset_btn = st.button("🔁 Reset session state")

if reset_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.success("Session state cleared. Reload the page.")

# Ensure a persistent working directory for this session
if "workdir" not in st.session_state:
    st.session_state.workdir = tempfile.mkdtemp(prefix="filings_")
workdir = st.session_state.workdir
db_dir = os.path.join(workdir, "chroma_db")
os.makedirs(db_dir, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def save_uploaded_files(uploaded_files) -> List[str]:
    """Save Streamlit UploadedFile objects to disk and return file paths."""
    fpaths = []
    for uf in uploaded_files or []:
        out_path = os.path.join(workdir, uf.name)
        with open(out_path, "wb") as f:
            f.write(uf.read())
        fpaths.append(out_path)
    return fpaths

def extract_tables_from_pdf(path: str) -> List[pd.DataFrame]:
    """Extract tables from a PDF file via pdfplumber."""
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tbls = page.extract_tables()
            for t in tbls:
                if not t or len(t) < 2:
                    continue
                # Make header safe and unique
                header = [str(h).strip() if h is not None else "" for h in t[0]]
                # Deduplicate empty or duplicate headers
                seen = {}
                for i, h in enumerate(header):
                    if h == "" or h in seen:
                        header[i] = f"{h or 'col'}_{i}"
                    seen[h] = True
                df = pd.DataFrame(t[1:], columns=header)
                # Drop fully empty rows
                df = df.dropna(how="all")
                if not df.empty:
                    out.append(df)
    return out

def normalize_numeric(val: str) -> float:
    """Parse currency/percent strings to float. Returns np.nan if not parseable."""
    if val is None:
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in {"na", "nan", "none", "-"}:
        return np.nan
    # Percent
    pct = re.search(r"(-?\d+(?:[\.,]\d+)?)\s*%", s)
    if pct:
        num = pct.group(1).replace(",", "")
        try:
            return float(num) / 100.0
        except:  # noqa: E722
            return np.nan
    # Currency / plain number with commas and parentheses for negatives
    s = s.replace(",", "")
    neg = False
    if "(" in s and ")" in s:
        neg = True
        s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[^\d\.\-]", "", s)  # remove non-numeric except . and -
    if s in {"", "-", ".", "-."}:
        return np.nan
    try:
        x = float(s)
        return -x if neg else x
    except:  # noqa: E722
        return np.nan

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].apply(normalize_numeric)
    return out

YEAR_RX = re.compile(r"(20\d{2}|19\d{2})")  # basic year matcher

def detect_year_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (year_like_columns, value_columns)."""
    year_cols = []
    for c in df.columns:
        if YEAR_RX.search(str(c)):
            year_cols.append(c)
    value_cols = [c for c in df.columns if c not in year_cols]
    return year_cols, value_cols

KPI_PATTERNS = {
    "Revenue": r"\brevenue|sales|turnover\b",
    "Net Income": r"\bnet\s+(income|profit|earnings)\b",
    "EPS": r"\b(eps|earnings per share)\b",
    "Operating Income": r"\boperating\s+income\b",
    "Total Assets": r"\btotal\s+assets\b",
    "Total Liabilities": r"\btotal\s+liabilities\b",
    "Cash Flow from Operations": r"\bcash\s*flow.*(operations|operating)\b",
}

def find_kpi_tables(tables: List[pd.DataFrame]) -> Dict[str, List[pd.DataFrame]]:
    kpi_map = {k: [] for k in KPI_PATTERNS}
    for df in tables:
        hay = " ".join(df.columns.astype(str)) + " " + " ".join(df.astype(str).fillna("").values.flatten())
        for k, pat in KPI_PATTERNS.items():
            if re.search(pat, hay, re.IGNORECASE):
                kpi_map[k].append(df)
    # Remove empties
    return {k: v for k, v in kpi_map.items() if v}

def to_excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in dfs.items():
            safe = re.sub(r"[^A-Za-z0-9_ ]", "_", name)[:31] or "Sheet"
            df.to_excel(writer, index=False, sheet_name=safe)
    return bio.getvalue()

# ----------------------------
# File Upload
# ----------------------------
uploaded_files = st.file_uploader("Upload Financial Filings (PDF)", type=["pdf"], accept_multiple_files=True)
pdf_paths = save_uploaded_files(uploaded_files)

# ----------------------------
# Ingest Text (RAG)
# ----------------------------
docs = []
if pdf_paths:
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())

if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": int(top_k)})
else:
    retriever = None

# ----------------------------
# Extract Tables
# ----------------------------
all_tables: Dict[str, List[pd.DataFrame]] = {}
if pdf_paths:
    with st.spinner("Extracting tables from PDFs..."):
        for p in pdf_paths:
            all_tables[p] = extract_tables_from_pdf(p)

# ----------------------------
# LLM (Groq) + QA Chain
# ----------------------------
qa_chain = None
if groq_api and retriever:
    llm = ChatGroq(api_key=groq_api,model=model_choice,temperature=0.2)
    template = """
You are a precise financial analysis assistant.
Use the retrieved context to answer. If you compute numbers, show them clearly with units (₹, $, €, %).
Be concise and factual. If unsure, say you don't know.
Context:
{context}
Question: {question}
"""
    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )

# ----------------------------
# UI: Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["💬 Q&A (Text RAG)", "📑 Tables & KPIs", "📈 Charts & Exports"])

# --- Tab 1: Q&A ---
with tab1:
    st.subheader("Ask questions about your filings")
    if qa_chain is None:
        st.info("Upload PDFs and enter your Groq API key to enable Q&A.")
    else:
        query = st.text_input("Your question")
        if query:
            with st.spinner("Thinking..."):
                resp = qa_chain(query)
            st.markdown("### 📌 Answer")
            st.write(resp["result"])
            with st.expander("Sources"):
                for i, src in enumerate(resp["source_documents"], 1):
                    st.write(f"**Source {i}**: " + src.page_content[:600] + ("..." if len(src.page_content) > 600 else ""))

# --- Tab 2: Tables & KPIs ---
with tab2:
    st.subheader("Extracted Tables")
    if not all_tables:
        st.info("Upload PDFs to see extracted tables.")
    else:
        for path, tables in all_tables.items():
            st.markdown(f"**📄 {os.path.basename(path)}** — {len(tables)} table(s)")
            kpi_map = find_kpi_tables(tables)
            if kpi_map:
                st.markdown("**Detected KPI Tables**")
                for kpi, tlist in kpi_map.items():
                    st.markdown(f"- **{kpi}**: {len(tlist)} table(s)")

            for idx, df in enumerate(tables, 1):
                st.markdown(f"**Table {idx}**")
                st.dataframe(df)

# --- Tab 3: Charts & Exports ---
with tab3:
    st.subheader("Trends & Downloads")

    if not all_tables:
        st.info("No tables to analyze yet.")
    else:
        # Combine KPI-first tables into named frames (first good table per KPI)
        first_kpi_frames: Dict[str, pd.DataFrame] = {}
        for path, tables in all_tables.items():
            kmap = find_kpi_tables(tables)
            for kpi, frames in kmap.items():
                if kpi not in first_kpi_frames and len(frames) > 0:
                    first_kpi_frames[kpi] = frames[0]  # take the first

        # Try to plot for KPI tables with year-like columns
        if first_kpi_frames:
            st.markdown("### 📈 Auto-Charts (if year columns detected)")

            for kpi, df in first_kpi_frames.items():
                # Coerce copy to numeric where possible
                df_num = df.copy()
                # If a typical "row label" column exists, set as index
                if df_num.columns.size > 1:
                    # Attempt to use first column as index (label)
                    df_num = df_num.set_index(df_num.columns[0])

                # Identify year columns
                year_cols, _ = detect_year_columns(df_num.reset_index())
                year_cols = [c for c in year_cols if c in df_num.columns]

                if year_cols:
                    # Build a 1D series across years by summing numeric values per year
                    df_coerced = df_num[year_cols].applymap(normalize_numeric)
                    yearly = df_coerced.sum(axis=0, skipna=True)
                    # Plot
                    try:
                        st.markdown(f"**{kpi} — Trend**")
                        fig, ax = plt.subplots(figsize=(6, 3))
                        yearly.index = yearly.index.astype(str)
                        ax.plot(yearly.index, yearly.values, marker="o")
                        ax.set_xlabel("Year")
                        ax.set_ylabel(kpi)
                        ax.set_title(f"{kpi} Trend")
                        ax.grid(True, linestyle="--", alpha=0.4)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Couldn't plot {kpi}: {e}")

        # ---- Exports ----
        st.markdown("### ⬇️ Export Tables")
        # Flatten all tables into a dict for downloads
        all_named_tables: Dict[str, pd.DataFrame] = {}
        count = 1
        for path, tables in all_tables.items():
            base = os.path.splitext(os.path.basename(path))[0]
            for df in tables:
                name = f"{base}_table_{count}"
                all_named_tables[name] = df
                count += 1

        if not all_named_tables:
            st.info("No tables to export.")
        else:
            # Export to single Excel (multi-sheet)
            xlsx_bytes = to_excel_bytes(all_named_tables)
            st.download_button(
                label="📘 Download ALL tables as Excel (.xlsx)",
                data=xlsx_bytes,
                file_name=f"financial_tables_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Export each as CSV in a single zipped file (optional simple CSV one-by-one too)
            # (Simpler: let users pick one table below)
            st.markdown("#### Export a single table as CSV")
            pick = st.selectbox("Choose a table", options=list(all_named_tables.keys()))
            if pick:
                csv_bytes = all_named_tables[pick].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"🧾 Download `{pick}.csv`",
                    data=csv_bytes,
                    file_name=f"{pick}.csv",
                    mime="text/csv",
                )

# ----------------------------
# Footer / Notes
# ----------------------------
st.markdown(
    """
---
**Notes / Guardrails**
- Numeric parsing handles `$`, `₹`, commas, parentheses for negatives, and `%`.
- If no year columns are detected, charts may not render; tables remain downloadable.
- For best results, upload native text PDFs (scanned images may require OCR first).
"""
)
