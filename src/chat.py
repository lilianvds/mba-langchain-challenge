import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from search import buscar_contexto

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower() # Default to gemini

if LLM_PROVIDER == "gemini":
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("A variável GOOGLE_API_KEY não está configurada no .env para o provedor Gemini.")
    LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
elif LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("A variável OPENAI_API_KEY não está configurada no .env para o provedor OpenAI.")
    LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo") # gpt-5-nano is not a real model, using gpt-3.5-turbo as a placeholder
else:
    raise ValueError("LLM_PROVIDER deve ser 'gemini' ou 'openai'.")

# Instancia a LLM exigida no desafio
# Usamos temperature=0 para que o modelo seja estritamente factual e não invente respostas
print(f"⚙️ Usando o provedor de LLM: {LLM_PROVIDER} com modelo de chat: {LLM_MODEL}")
if LLM_PROVIDER == "gemini":
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
else: # openai
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


# Template de prompt exatamente como exigido no desafio
PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

print(f"⚙️ Usando o provedor de LLM: {LLM_PROVIDER} com modelo de chat: {LLM_MODEL}")
if LLM_PROVIDER == "gemini":
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
else:  # openai
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


def _get_user_input() -> str:
    """Solicita a pergunta do usuário no terminal."""
    return input("PERGUNTA: ")


def _handle_exit_commands(query: str) -> bool:
    """Verifica se o usuário digitou um comando para sair."""
    if query.strip().lower() in ["sair", "exit", "quit"]:
        print("Encerrando o sistema...")
        return True
    return False


def _retrieve_context(query: str, k: int = 10) -> str:
    """Busca os chunks mais relevantes no banco de dados vetorial."""
    try:
        context = retrieve_context(query, k=k)
        if not context.strip():
            print(
                "RESPOSTA: Não tenho informações necessárias para responder sua pergunta, pois nenhum contexto relevante foi encontrado.\n"
            )
            return ""
        return context
    except Exception as e:
        print(f"Erro ao buscar contexto no banco de dados: {e}\n")
        return ""


def _format_llm_prompt(context: str, query: str) -> str:
    """Formata o prompt para a LLM com o contexto e a pergunta."""
    return PROMPT_TEMPLATE.format(contexto=context, pergunta=query)


def _get_llm_response(formatted_prompt: str):
    """Envia o prompt para a LLM e retorna a resposta."""
    return llm.invoke(formatted_prompt)


def _print_llm_response(response):
    """Imprime a resposta da LLM no terminal."""
    print(f"RESPOSTA: {response.content}\n")


def main():
    print("🤖 CLI de Busca Semântica (RAG) iniciado.")
    print("Digite 'sair' para encerrar a aplicação.\n")

    while True:
        try:
            pergunta = _get_user_input()
            if _handle_exit_commands(pergunta):
                break
            if not pergunta.strip(): # Ignora perguntas vazias
                continue

            contexto_banco = _retrieve_context(pergunta)
            if not contexto_banco: # Se não houver contexto, já foi impressa a mensagem
                continue

            prompt_formatado = _format_llm_prompt(contexto_banco, pergunta)
            resposta = _get_llm_response(prompt_formatado)
            _print_llm_response(resposta)

        except KeyboardInterrupt:
            print("\nEncerrando o sistema...")
            break
        except Exception as e:
            print(f"Ocorreu um erro: {e}\n")

if __name__ == "__main__":
    main()