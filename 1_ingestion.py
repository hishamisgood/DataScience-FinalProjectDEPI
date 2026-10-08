from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_document_chunks(data_folder_path="./data"):
    """
    تحميل ملفات PDF من المجلد وتقطيعها إلى Chunks.
    """
    # إعداد محمل الملفات لقراءة ملفات PDF
    loader = DirectoryLoader(
        data_folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # إعداد مقطع النصوص بالأرقام المطلوبة في المشروع
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    
    # تقطيع المستندات وإرجاع النتيجة
    chunks = text_splitter.split_documents(documents)
    
    return chunks