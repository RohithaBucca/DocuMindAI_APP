
import os

from functools import lru_cache
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq


load_dotenv()


@lru_cache(maxsize=1)
def load_embedding_model():

    model_name = os.getenv("MODEL_NAME")

    if not model_name:
        raise ValueError("MODEL_NAME is missing in .env")

    model = SentenceTransformer(model_name)

    return model



@lru_cache(maxsize=1)
def load_llm():

    groq_api = os.getenv("GROQ_API")
    model_name = os.getenv("LLM_MODEL")

    if not groq_api:
        raise ValueError("GROQ_API is missing")

    if not model_name:
        raise ValueError("LLM_MODEL is missing")

    temperature = float(
        os.getenv("LLM_TEMPERATURE", "0.3")
    )

    llm = ChatGroq(
        model=model_name,
        api_key=groq_api,
        temperature=temperature,

        # Limit the maximum generated tokens
        max_tokens=700
    )

    return llm
