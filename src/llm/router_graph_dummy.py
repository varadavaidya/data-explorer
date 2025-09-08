from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3:latest", temperature=0)

def run_router_graph(prompt: str):
    try:
        response = llm.invoke(prompt).content
        return {"kind": "text", "payload": response}, None, None
    except Exception as e:
        return None, None, str(e)
