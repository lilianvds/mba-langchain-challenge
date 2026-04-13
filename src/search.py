import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from config import DB_URL, LLM_PROVIDER, EMBEDDING_MODEL, COLLECTION_NAME


def _get_embeddings_model():
    """Returns the configured embeddings model (Gemini or OpenAI)."""
    print(f"⚙️ Using LLM provider: {LLM_PROVIDER} with embedding model: {EMBEDDING_MODEL}")
    if LLM_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    else:  # openai
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _initialize_vector_store():
    """Inicializa a conexão com o banco vetorial."""
    embeddings = _get_embeddings_model()
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )


def _perform_similarity_search(vector_store: PGVector, query: str, k: int):
    """Performs a similarity search in the vector store."""
    return vector_store.similarity_search_with_score(query, k=k)


def _format_retrieved_chunks(results) -> str:
    """Formats the retrieved chunks into a single string."""
    chunks = [doc.page_content for doc, score in results]
    return "\n\n".join(chunks)


def retrieve_context(query: str, k: int = 10) -> str:
    """
    Vectorizes the query and searches for the 'k' most relevant results in the database.
    Retorna o texto concatenado dos chunks encontrados.
    """
    vector_store = _initialize_vector_store()
    results = _perform_similarity_search(vector_store, query, k)
    context = _format_retrieved_chunks(results)
    return context