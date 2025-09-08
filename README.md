# 📊 Data Explorer

**Data Explorer** is a conversational web app (built with [Streamlit](https://streamlit.io)) that lets you:
- Upload CSV files
- Ask natural-language questions (e.g., *“Top 5 categories by revenue”*)
- Get answers as tables or charts
- Track your analysis history
- Save plots and queries into a local database (SQLite)

This project demonstrates:
- Multi-turn conversations with context memory
- Integration of **Pandas** for analysis
- **Matplotlib** for visualization
- **SQLite** for persistence (sessions, datasets, plots, saved views)
- A clean, modular structure for scalability

---

## Features
- **Upload CSVs** from the sidebar
- **Schema preview**: view column names and data types
- **Natural queries**: “What columns do I have?”, “Plot revenue by month”
- **Charts**: bar, line, grouped bar (saved into `artifacts/plots/`)
- **Persistence**: chats, datasets, and plots stored in `artifacts/app.db`
- **Extensible**: more tools, DSL queries, and advanced charts can be added

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/varadavaidya/data-explorer.git
cd data-explorer 
```


### 2. Create & activate venv 

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash 
pip install -r requirements.txt
```

### 4. Run OLLAMA's llama3

```bash
ollama list # if empty do below, else ignore
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama run llama3
```

### 5. Run the app 

```bash
streamlit run streamlit_app.py
```

---

### 5. (Optional) Creative Mode with LLM

Data Explorer supports an optional **Creative Mode**, powered by [Ollama](https://ollama.com/), that lets you phrase queries naturally:

- "minimum of Year"
- "most common vehicle types"
- "plot average range over time"
- "compare resale value by region"

Behind the scenes:
- A local LLM (Llama 3, Mistral, etc.) interprets your request.
- The app validates the JSON intent.
- The same safe Pandas/Matplotlib/SQLite tools execute the result.

### 6. To use Creative Mode:
1. Make sure [Ollama](https://ollama.com/download) is running.
2. Confirm you’ve pulled a model (e.g.):
   ```bash
   ollama pull llama3


### 📂 Project Structure

```
data-explorer/
├─ src/               # backend logic (memory, tools, db, nodes, graph)
├─ artifacts/         # runtime outputs (db, plots)
├─ examples/          # sample CSVs
├─ streamlit_app.py   # Streamlit frontend
├─ app.py             # CLI version
├─ requirements.txt   # dependencies
├─ .gitignore
└─ README.md
```
