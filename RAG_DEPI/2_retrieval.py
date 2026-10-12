from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "company_documents"
NUMBER_OF_CHUNKS_TO_RETRIEVE = 6


def build_retriever(chunks, persist_directory=DEFAULT_PERSIST_DIRECTORY):
    """Store the chunks in a Chroma vector store and return a retriever."""
    if not chunks:
        raise ValueError(
            "build_retriever received an empty list of chunks. "
            "Check the output of 1_ingestion.py."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    vector_store.reset_collection()
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": NUMBER_OF_CHUNKS_TO_RETRIEVE}
    )
    return retriever

if __name__ == "__main__":
    from langchain_core.documents import Document

    sample_chunks = [
        Document(
            page_content="Employees may work remotely up to three days per week.",
            metadata={"source": "sample_policy.pdf", "page": 1},
        ),
        Document(
            page_content="Expense reports must be submitted within 30 days.",
            metadata={"source": "sample_finance.pdf", "page": 2},
        ),
        Document(
            page_content="Sam Example joined the design team in 2021.",
            metadata={"source": "sample_people.pdf", "page": 3},
        ),
    ]

    test_retriever = build_retriever(sample_chunks, persist_directory="chroma_test_db")
    results = test_retriever.invoke("How many days can staff work from home?")

    for document in results:
        print(document.metadata, "->", document.page_content)