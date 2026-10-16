
from pathlib import Path
import importlib.util

from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# تحديد مسارات المشروع
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


# تحميل ملفات المراحل السابقة
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingestion = load_module(
    "ingestion",
    BASE_DIR / "1_ingestion.py"
)

retrieval = load_module(
    "retrieval",
    BASE_DIR / "2_retrieval.py"
)


# إنشاء QA Chain
def build_qa_chain():

    # Phase 1: قراءة ملفات PDF وتقسيم النص
    chunks = ingestion.get_document_chunks(
        data_folder_path=str(BASE_DIR / "data")
    )

    if not chunks:
        raise ValueError(
            "No PDF chunks found. Check RAG_DEPI/data."
        )

    # Phase 2: استرجاع الأجزاء المناسبة
    retriever = retrieval.build_retriever(
        chunks,
        persist_directory=str(PROJECT_DIR / "chroma_db")
    )

    # Phase 3: إعداد نموذج اللغة
    llm = OllamaLLM(model="llama3.2")

    # ربط الاسترجاع بتوليد الإجابات
   
    
    prompt_template = """
    You are a helpful assistant answering questions about company documents.

    Instructions:
    - Answer using only the provided context.
    - Give the direct answer in 3 to 5 bullet points.
    - Focus only on the requirements asked in the question.
    - Do not include unrelated security rules or extra policy details.
    - Do not repeat information.
    - Do not invent facts.
    - If the answer is not in the context, say:
      "I could not find the answer in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """


    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain



# تشغيل النظام والتفاعل مع المستخدم
if __name__ == "__main__":

    qa_chain = build_qa_chain()

    print("\n=== Company Documents RAG Assistant ===")
    print("Ask me anything about the company documents.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        # إنهاء البرنامج
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # منع إرسال سؤال فارغ
        if not question:
            print("Please enter a question.\n")
            continue

        # توليد الإجابة
        response = qa_chain.invoke({"query": question})

        print("\nAnswer:")
        print(response["result"])

        # عرض المصادر بدون تكرار نفس الملف والصفحة
        print("\nSource documents:")

        seen_sources = set()

        for document in response["source_documents"]:
            source = document.metadata.get("source", "Unknown")
            page = document.metadata.get(
                "page_label",
                document.metadata.get("page", "Unknown")
            )

            source_key = (source, page)

            if source_key not in seen_sources:
                seen_sources.add(source_key)

                print(f"File: {source}")
                print(f"Page: {page}")
                print("-" * 40)

        print()
