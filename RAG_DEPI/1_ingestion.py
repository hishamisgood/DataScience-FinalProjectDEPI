from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_document_chunks(data_folder_path="./data"):
    loader = DirectoryLoader(
        data_folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()

    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Adjusted for better semantic context retention
        chunk_overlap=50,
    )

    chunks = text_splitter.split_documents(documents)

    # Retained the validation check, but removed the redundant reassignment
    for chunk in chunks:
        if "source" not in chunk.metadata or "page" not in chunk.metadata:
            raise ValueError(
                "A document chunk is missing its source or page metadata."
            )

    return chunks
