from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function(
    ollama_embedding_model: str,
    region_name: str,
    use_bedrock: bool = False,
) -> OllamaEmbeddings | BedrockEmbeddings:
    """
    Get the embedding function based on the model and whether to use Bedrock.
    """
    if use_bedrock:
        return BedrockEmbeddings(
                                 credentials_profile_name="default",
                                 region_name=region_name
        )
    else:
        return OllamaEmbeddings(model=ollama_embedding_model,)
