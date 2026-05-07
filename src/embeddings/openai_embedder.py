from openai import OpenAI
import numpy as np
import os


def get_client():
    """
    Create OpenAI client AFTER environment variables are loaded
    """
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> np.ndarray:
    client = get_client()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )


def embed_query(query: str) -> np.ndarray:
    client = get_client()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    return np.array(response.data[0].embedding, dtype="float32")
