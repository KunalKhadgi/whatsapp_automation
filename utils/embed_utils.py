import numpy as np
import faiss
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import hashlib

client = OpenAI()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
embedding_cache = {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(Exception))
def embed_texts(texts):
    if len(texts) == 1:
        key = hashlib.md5(texts[0].encode()).hexdigest()
        if key in embedding_cache:
            return embedding_cache[key]
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    if len(texts) == 1:
        embedding_cache[key] = vecs
    return vecs
