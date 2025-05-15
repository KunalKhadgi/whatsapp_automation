# agents/summarize_agent.py

from openai import OpenAI
import tiktoken
import numpy as np

client = OpenAI()
MODEL           = "gpt-3.5-turbo"
WINDOW_TOKENS   = 4096
RESERVED_TOKENS = 500      # room for the reply
MAX_CHUNK_TOKENS = WINDOW_TOKENS - RESERVED_TOKENS

def count_tokens(text: str, model=MODEL) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def split_into_chunks(text: str, max_tokens: int, model=MODEL):
    """
    Splits `text` into chunks of <= max_tokens by token count.
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks

def run_summarization(params: dict, metas: list[dict]) -> dict:
    # --- 1) Build transcript ---
    target = params["target"]
    count  = params.get("count")
    chat_msgs = [m for m in metas if m["phone"] == target]
    chat_msgs.sort(key=lambda m: m["timestamp_start"])
    if count:
        chat_msgs = chat_msgs[-count:]

    transcript = "\n".join(f"{m['timestamp_start']} | {m['text']}"
                           for m in chat_msgs)

    # --- 2) Chunk if over window ---
    total_tokens = count_tokens(transcript)
    if total_tokens <= MAX_CHUNK_TOKENS:
        chunks = [transcript]
    else:
        chunks = split_into_chunks(transcript, MAX_CHUNK_TOKENS)

    # --- 3) Summarize each chunk in one LLM call each ---
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        prompt = (
            f"Summarize this portion of a WhatsApp conversation:\n```text\n"
            f"{chunk}\n```"
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system", "content":"You are a helpful assistant that summarizes conversations."},
                {"role":"user",   "content":prompt}
            ],
            max_tokens=RESERVED_TOKENS
        )
        chunk_summaries.append(resp.choices[0].message.content)

    # --- 4) Combine chunk summaries into one final summary ---
    if len(chunk_summaries) == 1:
        summary = chunk_summaries[0]
    else:
        combined_prompt = (
            "Combine the following partial summaries of the same conversation into one coherent summary:\n```\n"
            + "\n---\n".join(chunk_summaries)
            + "\n```"
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system", "content":"You are a helpful assistant that merges conversation summaries."},
                {"role":"user",   "content":combined_prompt}
            ],
            max_tokens=RESERVED_TOKENS
        )
        summary = resp.choices[0].message.content

    return {
        "summary": summary,
        "chunks": len(chunks),
        "original_tokens": total_tokens,
        "chunk_sizes": [count_tokens(c) for c in chunks]
    }
