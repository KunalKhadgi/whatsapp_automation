from tqdm import tqdm
from pathlib import Path
import re, os
import json
import numpy as np
from datetime import datetime
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')
DIM = int(os.getenv('FAISS_DIM', '1536'))
CHUNK_SIZE = 7
STRIDE = 6
SHORT_LIMIT = 7

client = OpenAI()

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

def parse_chat_txt(txt_path, phone):
    # Returns list of dicts: {timestamp, sender, message, direction}
    msgs = []
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*?): (.*)$')
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()
    buffer = []
    for line in lines:
        line = line.rstrip('\n')
        m = pattern.match(line)
        if m:
            # Save previous message
            if buffer:
                msgs.append(buffer[-1])
            ds, ts, snd, msg = m.groups()
            try:
                dt = datetime.strptime(f"{ds} {ts}", "%d/%m/%Y %H:%M")
                timestamp = dt.isoformat()
            except ValueError:
                timestamp = f"{ds}T{ts}"
            # Infer direction: outgoing if sender is 'me' or phone, else incoming
            direction = "outgoing" if snd.strip().lower() in ["me", phone] else "incoming"
            buffer = [{
                "timestamp": timestamp,
                "sender": snd,
                "message": msg,
                "direction": direction
            }]
        else:
            # Continuation of previous message
            if buffer:
                buffer[-1]["message"] += "\n" + line
    if buffer:
        msgs.append(buffer[-1])
    # Filter out system messages
    msgs = [m for m in msgs if not m["sender"].lower().startswith("messages to this group")]
    return msgs

def get_chat_type(msgs):
    # If more than 2 unique senders, treat as group
    senders = set(m["sender"] for m in msgs)
    return "group" if len(senders) > 2 else "individual"

def main():
    data_dir = Path("processed_exported_data")
    faiss_dir = Path("faiss_db")
    faiss_dir.mkdir(exist_ok=True)
    index = faiss.IndexHNSWFlat(DIM, 32)
    metas = []

    for chat_folder in tqdm(list(data_dir.iterdir()), desc="Chats"):
        if not chat_folder.is_dir():
            continue
        phone = chat_folder.name
        txt_path = chat_folder / f"{phone}.txt"
        if not txt_path.exists():
            continue
        msgs = parse_chat_txt(txt_path, phone)
        n = len(msgs)
        if n == 0:
            continue
        print(f"\nProcessing chat: {phone} ({n} messages)")
        chat_type = get_chat_type(msgs)
        # Short chat: embed each message
        if n < CHUNK_SIZE:
            for i, m in enumerate(msgs):
                text = f"{m['sender']}: {m['message']}"
                preview = text[:80].replace('\n', ' ')
                print(f"  Embedding single message {i+1}/{n}: {preview}...")                
                emb = embed_texts([text])
                index.add(emb)
                metas.append({
                    "uid": f"{phone}|{i}|single",
                    "phone": phone,
                    "start_idx": i,
                    "end_idx": i+1,
                    "timestamp_start": m["timestamp"],
                    "timestamp_end": m["timestamp"],
                    "text": text,
                    "raw_messages": [m],
                    "sender_list": [m["sender"]],
                    "direction_list": [m["direction"]],
                    "chat_type": chat_type
                })
        # Long chat: sliding window
        else:
            for start in range(0, n, STRIDE):
                chunk = msgs[start:start+CHUNK_SIZE]
                if not chunk:
                    break
                text = "\n".join(f"{m['sender']}: {m['message']}" for m in chunk)
                preview = text[:80].replace('\n', ' ')
                print(f"  Embedding chunk {start}-{start+len(chunk)}: {preview}...")
                emb = embed_texts([text])
                index.add(emb)
                metas.append({
                    "uid": f"{phone}|{start}-{start+len(chunk)}",
                    "phone": phone,
                    "start_idx": start,
                    "end_idx": start+len(chunk),
                    "timestamp_start": chunk[0]["timestamp"],
                    "timestamp_end": chunk[-1]["timestamp"],
                    "text": text,
                    "raw_messages": chunk,
                    "sender_list": list({m["sender"] for m in chunk}),
                    "direction_list": [m["direction"] for m in chunk],
                    "chat_type": chat_type
                })
    # Save index and metas
    faiss.write_index(index, str(faiss_dir / "whatsapp.index"))
    np.save(str(faiss_dir / "whatsapp_metas.npy"), np.array(metas, object), allow_pickle=True)
    print(f"Rebuilt vector DB with {len(metas)} chunks/messages.")

if __name__ == "__main__":
    main() 