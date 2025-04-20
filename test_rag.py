from query_data import query_rag
from langchain_ollama import OllamaLLM


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false')Does the actual response match the expected resonse?
"""


def test_rules():
    assert query_and_evaluate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def query_and_evaluate(
    question: str,
    expected_response: str,
) -> bool:
    """Queries the RAG system and evaluates the response."""
    # Query the RAG system
    response_text = query_rag(question)

    # Evaluate the response
    evaluation_prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text,
    )

    model = OllamaLLM(
        model="mistral",
    )
    evaluation_result_str = model.invoke(evaluation_prompt)
    evaluation_result_str_cleaned = evaluation_result_str.strip().lower()

    print(evaluation_prompt)

    if "true" in evaluation_result_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_result_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_result_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_result_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Unexpected evaluation response: {evaluation_result_str_cleaned}"
        )