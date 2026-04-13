import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    DB_URL,
    LLM_PROVIDER,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def _validate_pdf_path(pdf_path: str):
    """Valida se o arquivo PDF existe no caminho especificado."""
    if not os.path.exists(pdf_path):
        print(f"Erro: O arquivo '{pdf_path}' não foi encontrado.")
        sys.exit(1)
    print(f"📄 Iniciando a leitura do arquivo: {pdf_path}")


def _load_pdf_documents(pdf_path: str) -> list[Document]:
    """Carrega o conteúdo do PDF usando PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ PDF carregado. Total de páginas: {len(documents)}")
    return documents


def _split_documents_into_chunks(documents: list[Document]) -> list[Document]:
    """Divide os documentos em chunks de texto."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks: list[Document] = text_splitter.split_documents(documents)
    print(f"✂️ Texto dividido em {len(chunks)} chunks.")
    return chunks


def _get_embeddings_model():
    """Retorna o modelo de embeddings configurado (Gemini ou OpenAI)."""
    print(
        f"⚙️ Usando o provedor de LLM: {LLM_PROVIDER} com modelo de embedding: {EMBEDDING_MODEL}"
    )
    if LLM_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    else:  # openai
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _ingest_chunks_to_vector_store(chunks: list[Document], embeddings):
    """Conecta ao PostgreSQL e insere os chunks vetorizados."""
    print("💾 Conectando ao banco de dados e gerando os embeddings...")

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )
    try:
        vector_store.add_documents(chunks)
    except Exception as e:
        print(f"Erro ao inserir documentos no banco de dados: {e}")
        sys.exit(1)
    print("🚀 Ingestão concluída com sucesso! Os dados estão no PostgreSQL.")


def main():
    pdf_path = "document.pdf"
    _validate_pdf_path(pdf_path)
    documents = _load_pdf_documents(pdf_path)
    chunks = _split_documents_into_chunks(documents)
    embeddings = _get_embeddings_model()
    _ingest_chunks_to_vector_store(chunks, embeddings)

if __name__ == "__main__":
    main()