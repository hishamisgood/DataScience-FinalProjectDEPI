from pathlib import Path
import importlib.util
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

def load_generation_module():
    generation_path = BASE_DIR / "3_generation.py"
    spec = importlib.util.spec_from_file_location("generation", generation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {generation_path}")
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    return generation

@st.cache_resource
def get_qa_chain():
    generation = load_generation_module()
    return generation.build_qa_chain()

st.set_page_config(page_title="Company RAG Assistant", page_icon="📄")
st.title("📄 Company Documents RAG Assistant")
st.caption("Ask anything about internal company policies and personnel records.")

qa_chain = get_qa_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about company documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            response = qa_chain.invoke({"query": user_query})
            answer = response["result"]

            seen_sources = set()
            sources_list = []
            for doc in response.get("source_documents", []):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page_label", doc.metadata.get("page", "Unknown"))
                source_key = (source, page)
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources_list.append(f"- **File:** `{source}` | **Page:** `{page}`")

            if sources_list:
                full_output = f"{answer}\n\n---\n**Source Documents:**\n" + "\n".join(sources_list)
            else:
                full_output = answer

            st.markdown(full_output)

    st.session_state.messages.append({"role": "assistant", "content": full_output})