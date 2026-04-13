# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de **Ingestão e Busca Semântica em PDFs** utilizando RAG (Retrieval-Augmented Generation). Esta solução transforma documentos estáticos em uma base de conhecimento consultável via linha de comando (CLI).

##  Tecnologias utilizadas:

Orquestração do fluxo de RAG e integração com LLMs.
* **PostgreSQL + pgVector:** Armazenamento do banco de dados vetorial.
* **Docker & Docker Compose:** Containerização do ambiente de dados.
* **Python:** Linguagem principal da aplicação.

##  Pré-requisitos ambiente de desenvolvimento

Antes de começar, certifique-se de ter instalado em sua máquina:
* [Docker](https://docs.docker.com/get-docker/) e Docker Compose
* Python 3.10 ou superior
* Git

##  Como executar a solução

### 1. Clonar o repositório
```bash
git clone git@github.com:lilianvds/mba-langchain-challenge.git
cd mba-langchain-challenge

### 2. Iniciar o Banco de Dados (PostgreSQL com pgVector)
Utilizamos Docker Compose para subir o banco de dados. Certifique-se de ter o Docker instalado e rodando.
```bash
docker compose up -d
```

### 3. Configurar o Ambiente Python
Crie e ative um ambiente virtual Python, e instale as dependências do projeto:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configurar as Variáveis de Ambiente
Crie um arquivo chamado `.env` na raiz do projeto. Ele deve conter a string de conexão do banco local e as chaves das APIs dos LLMs que serão utilizados.

```bash
# Configuração do provedor de LLM (gemini ou openai)
LLM_PROVIDER=gemini # ou openai

DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# Variáveis para Gemini (se LLM_PROVIDER estiver configurado como 'gemini')
GOOGLE_API_KEY=sua_chave_do_google_aqui
GEMINI_EMBEDDING_MODEL=models/text-embedding-004 
GEMINI_LLM_MODEL=gemini-1.5-flash-lite

# Variáveis para OpenAI (se LLM_PROVIDER=openai)
OPENAI_API_KEY=sk-sua_chave_da_api_aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-3.5-turbo

##  Como Executar o Projeto

Antes de prosseguir, certifique-se de que o Docker com o PostgreSQL está rodando (verifique com `docker compose ps`).

### 1. Ingestão do PDF
Execute o script de ingestão para ler o `document.pdf`, dividir em chunks, gerar embeddings e armazená-los no banco de dados vetorial.
```bash
python3 src/ingest.py
```

### 2. Iniciar o Chat
Após a ingestão, você pode iniciar o chat interativo via CLI:
```bash
python3 src/chat.py