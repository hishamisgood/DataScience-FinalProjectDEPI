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
        chunk_size=400,
        chunk_overlap=50,
    )

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")

        if source is None or page is None:
            raise ValueError(
                "A document chunk is missing its source or page metadata."
            )

        chunk.metadata["source"] = source
        chunk.metadata["page"] = page

    return chunks
