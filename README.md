# financial-report-summarizer
AI-Powered Financial Report Summarizer using LLMs &amp; Streamlit
📊 AI-Powered Financial Report Summarizer

An interactive application that uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to analyze and summarize financial reports (10-K, 10-Q, balance sheets, income statements).

Built with LangChain, HuggingFace embeddings, ChromaDB, and Streamlit for financial Q&A, trend analysis, and structured export.

✨ Features

📂 Upload PDFs of financial filings (10-K, 10-Q, etc.)

🤖 LLM-powered Q&A for insights from balance sheets, cash flows, risk sections

🔍 RAG pipeline with ChromaDB + HuggingFace embeddings for accurate retrieval

📊 Interactive visualizations (revenue, expense trends, ratios)

📑 Export results to Excel/CSV for reporting

⚡ Groq LLM API integration for low-latency inference

🛠️ Tech Stack

Frontend: Streamlit

Backend: LangChain, Groq LLM API

NLP: HuggingFace Embeddings, RAG

Database: ChromaDB (Vector Store)

Data Extraction: PyPDFLoader (PDF parsing)

Visualization: Matplotlib / Streamlit Charts

Exports: Pandas (Excel/CSV)

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/avinash4724/financial-report-summarizer.git
cd financial-report-summarizer

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py

5️⃣ Enter API Key

Get your Groq API key: https://console.groq.com/keys

Paste it in the sidebar when running the app

📌 Example Use Cases

Summarize company 10-K filings in minutes

Extract revenue, expenses, risk factors for investor insights

Automate financial reporting workflows

Export insights to Excel for further analysis

🔮 Future Enhancements

Guardrails (currency formatting, numeric validation)

Async multi-file queries for faster processing

GraphDB integration for subsidiaries, segments, risk networks

Model fine-tuning for domain-specific accuracy
