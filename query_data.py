import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from read_config import read_config


CONFIG = read_config()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question best on the above conetext: {question}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Query the database with a question."
    )
    parser.add_argument("query_text", type=str, help="The question to ask.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the database.
    chroma_path = CONFIG["chroma_path"]

    embedding_function = get_embedding_function(
        ollama_embedding_model=CONFIG["embedding"]["ollama_embedding_model"],
        region_name=CONFIG["region_name"],
        use_bedrock=CONFIG["embedding"]["use_bedrock"],
    )
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function
        )

    # Search the database for relevant documents.
    results = db.similarity_search(query_text, k=CONFIG.get("top_k"))

    context_text = "\n\n---\n\n".join(
        [result.page_content for result in results]
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_prompt(
        context=context_text,
        question=query_text,
    )
    print(f"Prompt:\n{prompt}")

    model = OllamaLLM(
        model=CONFIG['llm']["model_name"]
    )
    response_text = model.invoke(prompt)

    sources = [
        result.metadata.get("id", "Unknown")
        for result in results
    ]
    formatted_response = (
        f"Response: {response_text}\n\n"
        f"Sources: {', '.join(sources)}"
    )
    print(f"Response:\n{formatted_response}")
    return formatted_response


if __name__ == "__main__":
    main()
