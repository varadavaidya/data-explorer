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

### 4. Run the app 

```bash
streamlit run streamlit_app.py
```

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
