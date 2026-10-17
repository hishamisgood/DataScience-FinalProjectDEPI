import importlib.util
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

question1 = [
    "What is Grace Okafor's key strenghts",
    "Can you help me with the best phone in the market?",
    "What is Rachel Kim's Goals?",
    "What is Hybrid in the company.",
    "If I had a baby, and I don't feel like working what should I do?",
]


def evaluate_questions(qa_chain: Any) -> dict[str, list[dict[str, str]]]:
    memory_log: dict[str, list[dict[str, str]]] = {"history": []}

    for question in question1:
        response = qa_chain.invoke({"query": question})
        memory_log["history"].append(
            {"question": question, "answer": response["result"]}
        )

    return memory_log


def build_qa_chain() -> Any:
    generation_path = BASE_DIR / "3_generation.py"
    spec = importlib.util.spec_from_file_location("generation", generation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generation module: {generation_path}")

    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    return generation.build_qa_chain()


def main() -> None:
    qa_chain = build_qa_chain()
    memory_log = evaluate_questions(qa_chain)
    print("=== Evaluation Results ===")
    print(json.dumps(memory_log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
