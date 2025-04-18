import argparse
import os
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from read_config import read_config

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Load the configuration file.
CONFIG = read_config()


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser(
        description="Populate the database with PDF documents."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Resetting the database...")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    print(
        f"Loaded {len(documents)} documents and split them into "
        f"{len(chunks)} chunks."
    )
    add_to_chroma(chunks)
    print("Database populated successfully.")


def load_documents() -> list[Document]:
    # Load PDF documents from the specified directory.
    data_path = CONFIG["data_path"]
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {data_path}.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["text_splitter"]["chunk_size"],
        chunk_overlap=CONFIG["text_splitter"]["chunk_overlap"],
        length_function=len,
        is_separator_regex=CONFIG["text_splitter"]["is_separator_regex"],
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Create a Chroma vector store and add the chunks to it.
    chroma_path = CONFIG["chroma_path"]

    embedding_function = get_embedding_function(
        ollama_embedding_model=CONFIG["embedding"]["ollama_embedding_model"],
        region_name=CONFIG["region_name"],
        use_bedrock=CONFIG["embedding"]["use_bedrock"],
    )

    # Create a new Chroma instance or load an existing one.
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function,
    )

    # Calculate Page IDs for the chunks.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the chunks in the Chroma database.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    print(f"Number of chunks to add: {len(chunks_with_ids)}")

    # Only add documents that don't already exist in the database.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) > 0:
        print(f" Adding new documents to the database: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(
            new_chunks,
            ids=new_chunk_ids,
        )
    else:
        print("No new documents to add to the database.")


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Calculate the page IDs for each chunk of text.
    This will create IDs like 'data/monopoly.pdf: 6:3'
    Page Source: Page Number : Chunk Index
    This is useful for tracking where each chunk came from in the original
    document.

    Args:
        chunks (list[Document]): A list of Document objects representing the
        text chunks.

    Returns:
        list[Document]: A list of Document objects with updated metadata
        containing the page IDs.
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            # If it's a new page, reset the index.
            current_chunk_index = 0
            last_page_id = current_page_id

        # Update the chunk metadata with the new page ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Update the chunk metadata with the new page ID.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    # Remove the existing database directory.
    chroma_path = CONFIG["chroma_path"]
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print(f"Removed existing database at {chroma_path}.")
    else:
        print(f"No existing database found at {chroma_path}.")


if __name__ == "__main__":
    main()
