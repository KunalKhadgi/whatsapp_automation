#!/usr/bin/env python3
"""
unified_app.py

– Unified WhatsApp processor + webhook → persists to TXT + FAISS
– /find_chat: exact chat retrieval by name/number
  /search: 
     • "summarize my chat with X" → last N messages, raw
     • "summarize the last K creators who confirmed they joined TOPIC" → custom logic
     • otherwise → semantic FAISS lookup
"""

import os, json, re, http.client, ssl, traceback
from datetime import datetime, timedelta
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytz
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, jsonify, Blueprint
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken
from dotenv import load_dotenv

# Add GCS download logic at startup
from google.cloud import storage

def download_from_gcs(bucket_name, blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {blob_name} to {destination_file_name}")
    else:
        print(f"Blob {blob_name} does not exist in bucket {bucket_name}.")

def download_directory_from_gcs(bucket_name, prefix, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        # Remove the prefix from the blob name to get the relative path
        rel_path = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")

def download_meta_jsons_from_gcs(bucket_name, prefix, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith("meta.json"):
            rel_path = os.path.relpath(blob.name, prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")

def download_from_gcs_if_missing(bucket_name, blob_name, destination_file_name):
    if os.path.exists(destination_file_name):
        print(f"Local file {destination_file_name} exists, skipping download.")
        return
    download_from_gcs(bucket_name, blob_name, destination_file_name)

# Download vector DB files at startup
BUCKET_NAME = "whatsapp-bot-2025-data"
download_from_gcs_if_missing(BUCKET_NAME, "faiss_db/whatsapp.index", "faiss_db/whatsapp.index")
download_from_gcs_if_missing(BUCKET_NAME, "faiss_db/whatsapp_metas.npy", "faiss_db/whatsapp_metas.npy")

# Load environment variables
load_dotenv()

from parser import bootstrap_conversation
# from parser import append_to_txt
# ─── CONFIG & SETUP ─────────────────────────────────────────────────────────

app           = Flask(__name__)
webhook_bp    = Blueprint("webhook", __name__)
TZ            = pytz.timezone(os.getenv('TIMEZONE', 'Asia/Kolkata'))

# Initialize OpenAI client
client = OpenAI()

# FAISS
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')
DIM         = int(os.getenv('FAISS_DIM', '1536'))
faiss_dir   = Path(os.getenv('FAISS_DIR', './faiss_db'))
faiss_dir.mkdir(exist_ok=True)
INDEX_PATH  = faiss_dir / "whatsapp.index"
METAS_PATH  = faiss_dir / "whatsapp_metas.npy"

# Chunking configuration
CHUNK_SIZE  = int(os.getenv('CHUNK_SIZE', '7'))    # messages per chunk
STRIDE      = int(os.getenv('STRIDE', '6'))        # sliding window stride
SHORT_LIMIT = int(os.getenv('SHORT_LIMIT', '4'))   # ≤ this many messages → embed individually

if INDEX_PATH.exists() and METAS_PATH.exists():
    index, metas = (
        faiss.read_index(str(INDEX_PATH), faiss.IO_FLAG_MMAP),
        np.load(str(METAS_PATH), allow_pickle=True).tolist()
    )
else:
    index, metas = faiss.IndexHNSWFlat(DIM, 32), []

faiss_lock  = Lock()
executor    = ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', '18')))

# Only download meta.json files for contact mapping
BUCKET_NAME = "whatsapp-bot-2025-data"
download_meta_jsons_from_gcs(BUCKET_NAME, "processed_exported_data/", "processed_exported_data/")

# Build contacts dict after meta.json files are downloaded
contacts = {}
processed_dir = Path(os.getenv('PROCESSED_DIR', './processed_exported_data'))
for d in processed_dir.iterdir():
    if not d.is_dir(): continue
    mf = d/"meta.json"
    if mf.exists():
        try:
            nm = json.loads(mf.read_text()).get("display_name","").strip().lower()
            if nm:
                contacts[nm] = d.name
                print(f"Loaded contact: {nm} -> {d.name}")
            else:
                print(f"meta.json in {d} missing display_name or is empty")
        except Exception as e:
            print(f"Error reading {mf}: {e}")
print(f"All loaded contacts: {contacts}")

# After loading metas, build an in-memory chat_map for fast lookup
from collections import defaultdict
chat_map = defaultdict(list)
for m in metas:
    chat_map[m["phone"]].append(m)

# ─── REGEX INTENTS ──────────────────────────────────────────────────────────

PHONE_RE          = re.compile(r"\b(\+?\d{10,14})\b")
SUMMARIZE_CHAT_RE = re.compile(
    r"""(?ix)                # case-insensitive, verbose
    \b
    (?:summarize|summary)    # verb
    (?:\s+my)?               # optional "my "
    \s+
    (?:chat|chats|conversation|conversations)
    (?:\s+(?:with|for))\s*
    (.+)$                     # capture target
    """
)

# Query intent detection using llm

def count_tokens(messages, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for m in messages:
        # Each message follows OpenAI's chat format
        num_tokens += 4  # every message metadata
        for key, value in m.items():
            num_tokens += len(enc.encode(value))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def extract_confirm_intent_with_llm(query: str) -> dict | None:
    """
    Uses the LLM to pull out:
      - the number of creators (count)
      - the topics they 'joined' (topics: list[str])
    from queries like:
      "summarize the last 5 creators who confirmed they joined discord and slack"

    Returns {"count":5, "topics":["discord","slack"]} or None.
    """
    prompt = f"""
Extract JSON from the user request.  If it is asking:
  "summarize the last 5 creators who confirmed they joined discord and slack"
you should output:

{{ 
  "count": 5,
  "topics": ["discord","slack"]
}}

If it is _not_ a "last N creators who joined TOPIC…" request, output EXACTLY:
NONE

Respond with *only* the JSON or NONE.
Now parse this:
"{query}"
"""
    messages = [
        {"role": "system", "content": "Return only valid JSON or NONE."},
        {"role": "user",   "content": prompt}
    ]
    print(f"[DEBUG] Tokens for intent detection: {count_tokens(messages)}")
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    print("[DEBUG] Raw LLM response:", resp.choices[0].message.content)
    text = resp.choices[0].message.content.strip()
    if text.upper() == "NONE":
        return None
    try:
        data = json.loads(text)
        # ensure topics is a list
        if isinstance(data.get("topics"), list) and isinstance(data.get("count"), int):
            return data
    except json.JSONDecodeError:
        pass
    return None


# ─── EMBEDDING UTILS ────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(),
       retry=retry_if_exception_type(Exception))
def _embed(texts:list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

def embed_and_persist(phone, sender, message, ts, direction):
    """
    Embeds and persists messages using the same chunking strategy as update_vectordb.py.
    For short chats (≤ SHORT_LIMIT messages), embeds each message individually.
    For longer chats, uses sliding windows of CHUNK_SIZE messages with STRIDE.
    """
    # Get all messages for this phone number
    chat_msgs = [m for m in metas if m["phone"] == phone]
    chat_msgs.sort(key=lambda m: m["timestamp"])
    
    # Add the new message
    chat_msgs.append({
        "timestamp": ts,
        "sender": sender,
        "message": message,
        "direction": direction
    })
    
    n = len(chat_msgs)
    
    # For short chats, embed each message individually
    if n <= SHORT_LIMIT:
        for i, m in enumerate(chat_msgs):
            uid = f"{phone}|{i}|single"
            if any(meta['uid'] == uid for meta in metas):
                continue
            text = f"{m['sender']}: {m['message']}"
            try:
                emb = _embed([text])
                with faiss_lock:
                    index.add(emb)
                    metas.append({
                        "uid": uid,
                        "phone": phone,
                        "start_idx": i,
                        "end_idx": i + 1,
                        "text": text
                    })
            except Exception as e:
                print(f"[ERR] {uid}: {e}")
                continue
    
    # For longer chats, use sliding windows
    else:
        for start in range(0, n, STRIDE):
            chunk = chat_msgs[start:start + CHUNK_SIZE]
            if not chunk:
                break
            uid = f"{phone}|{start}-{start + len(chunk)}"
            if any(meta['uid'] == uid for meta in metas):
                continue
            text = "\n".join(f"{c['sender']}: {c['message']}" for c in chunk)
            try:
                emb = _embed([text])
                with faiss_lock:
                    index.add(emb)
                    metas.append({
                        "uid": uid,
                        "phone": phone,
                        "start_idx": start,
                        "end_idx": start + len(chunk),
                        "text": text
                    })
            except Exception as e:
                print(f"[ERR] {uid}: {e}")
                continue
    
    # Persist changes
    with faiss_lock:
        faiss.write_index(index, str(INDEX_PATH))
        np.save(str(METAS_PATH), np.array(metas, object), allow_pickle=True)

# ─── MESSAGE PROCESSING ────────────────────────────────────────────────────

def process_message_event(raw:dict):
    body = (raw.get("body") or "").strip()
    if not body: return

    fm    = raw.get("fromMe",False)
    addr  = raw.get("to") if fm else raw.get("from")
    if not addr: return

    is_grp = addr.endswith("@g.us")
    chat_id = addr.replace("@c.us","").replace("@g.us","")
    display = raw.get("chatName") if is_grp else raw.get("pushname") or chat_id

    # determine sender
    if is_grp:
        author = raw.get("author","").split(":",1)[0].replace("@c.us","")
        sender = author or display
    else:
        sender = display

    # bootstrap folder & meta
    bootstrap_conversation(chat_id, display)

    # append to txt + embed
    ts_txt = datetime.now(TZ).strftime("%d/%m/%Y, %H:%M")
    # append_to_txt(chat_id,sender,body,timestamp=ts_txt)
    embed_and_persist(chat_id,sender,body,datetime.now(TZ).isoformat(),
                      "outgoing" if fm else "incoming")

# ─── HELPER LOOKUPS ─────────────────────────────────────────────────────────

def extract_phone(q:str)->str|None:
    m = PHONE_RE.search(q)
    return re.sub(r"\D","",m.group(1)) if m else None

def extract_contact(q:str)->tuple[str|None,str|None]:
    ql = q.lower()
    for name,ph in contacts.items():
        if ql in name:
            return ph,name
    return None,None

# ---- Function API ------------
filter_fn = {
    "name": "filter_confirmed_creators",
    "description": "Return the last K distinct creators whose incoming messages mention one of the given topics",
    "parameters": {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of creators to return"
            },
            "topics": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Keywords to look for in incoming messages"
            }
        },
        "required": ["count", "topics"]
    }
}

def filter_confirmed_creators(messages, topic: str, count: int):
    # your logic: scan `messages` for incoming "yes I joined <topic>" etc.
    creators = []
    for m in messages:
        if m['direction']=='incoming' and topic in m['message'].lower():
            creators.append(m['sender'])
    # take the last `count` unique creators
    return {"creators": list(dict.fromkeys(creators)[-count:])}


# ------- Function schemas and implementation -----------------


from flask import Flask, request, jsonify, Blueprint
from openai import OpenAI
import json

webhook_bp = Blueprint('webhook', __name__)

# --- 1) Function schemas + implementations ---

# A) Extract arbitrary intent parameters via LLM
EXTRACT_INTENT_FN = {
    "name": "extract_intent",
    "description": "Extract high-level intent (confirm-join, summarize-chat, retrieve-keyword) and parameters from the user query.",
    "parameters": {
      "type": "object",
      "properties": {
        "intent": {
          "type": "string",
          "enum": ["confirm_join", "summarize_chat", "keyword_search"]
        },
        "count":   {"type":"integer"},
        "topics":  {"type":"array","items":{"type":"string"}},
        "target":  {"type":"string"},
        "keyword": {"type":"string"}
      },
      "required": ["intent"]
    }
}

def extract_intent(query: str):
    # (unused in Python; purely driven by the model's function_call)
    pass

# B) Confirm-join filter
FILTER_CONFIRMED_FN = {
    "name": "filter_confirmed_creators",
    "description": "Given a list of messages, return last N creators who confirmed they joined specified topics.",
    "parameters": {
      "type":"object",
      "properties":{
        "topics": {"type":"array","items":{"type":"string"}},
        "count":  {"type":"integer"}
      },
      "required":["topics","count"]
    }
}
def filter_confirmed_creators(messages, topics, count):
    seen, output = set(), []
    for m in reversed(messages):
        if m['direction']=="incoming" and any(t in m['message'].lower() for t in topics):
            if m['sender'] not in seen:
                seen.add(m['sender'])
                output.append(m['sender'])
                if len(output)>=count:
                    break
    return {"creators": output}

# C) Summarize-chat retriever (raw last K msgs)
RETRIEVE_CHAT_FN = {
    "name": "retrieve_chat_history",
    "description": "Return last K messages from a given chat_id.",
    "parameters": {
      "type":"object",
      "properties":{
        "chat_id": {"type":"string"}, 
        "count":   {"type":"integer"}
      },
      "required":["chat_id","count"]
    }
}
def retrieve_chat_history(messages, chat_id, count):
    subset = [m for m in messages if m['phone']==chat_id]
    return {"history": subset[-count:]}

# D) Keyword search over vector DB
VECTOR_SEARCH_FN = {
    "name": "vector_keyword_search",
    "description": "Perform a FAISS vector-based semantic search over all messages for a keyword query.",
    "parameters": {
      "type":"object",
      "properties":{
        "query": {"type":"string"},
        "count": {"type":"integer"}
      },
      "required":["query","count"]
    }
}
def vector_keyword_search(query, count):
    """
    Perform a FAISS vector-based semantic search over all messages.
    Returns the most relevant chunks based on the query.
    """
    emb = _embed([query])
    d, idxs = index.search(emb, count * 5)  # Get more results than needed for deduplication
    results = []
    seen_phones = set()  # Track unique phone numbers
    
    for i in idxs[0]:
        if 0 <= i < len(metas):
            m = metas[i]
            # Skip if we've already seen this phone number
            if m['phone'] in seen_phones:
                continue
            seen_phones.add(m['phone'])
            results.append({
                "entry": m['text'],
                "phone": m['phone'],
                "start_idx": m['start_idx'],
                "end_idx": m['end_idx']
            })
            if len(results) >= count:
                break
    
    return {"results": results}


# ─── SEARCH AGENT SYSTEM ─────────────────────────────────────────────────────

class SearchAgent:
    """Agent system to handle different types of search queries"""
    
    def __init__(self):
        self.handlers = {
            "summarize_chat": self.handle_summarize_chat,
            "find_creators": self.handle_find_creators,
            "semantic_search": self.handle_semantic_search,
            "keyword_search": self.handle_keyword_search,
            "analyze_creators": self.handle_analyze_creators  # New handler
        }
    
    def detect_intent(self, query: str) -> tuple[str, dict]:
        print(f"[DEBUG] Step 3: Detecting intent for query: {query}")
        prompt = f"""
        Analyze this search query and determine its intent and parameters.
        Query: \"{query}\"

        Return a JSON object with:
        - intent: one of ["summarize_chat", "find_creators", "semantic_search", "keyword_search", "analyze_creators"]
        - params: relevant parameters for the intent

        Rules for parameter extraction:
        1. If the query is about finding people who discussed certain topics, use "semantic_search" intent
        2. If the query mentions a number, extract it as the "count" parameter
        3. For semantic search, use the topic as the "query" parameter
        4. If no count is mentioned, omit the count parameter to return all results

        Examples:
        - "who are the 20 people to whom I talked about discord" → {{"intent": "semantic_search", "params": {{"query": "discord", "count": 20}}}}
        - "find people who discussed slack" → {{"intent": "semantic_search", "params": {{"query": "slack"}}}}
        - "summarize my chat with John" → {{"intent": "summarize_chat", "params": {{"target": "John"}}}}

        Always return a JSON object with the correct fields. Do not include any text or explanation.
        """

        messages = [
            {"role": "system", "content": "You are a query intent analyzer. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        print(f"[DEBUG] Tokens for intent detection: {count_tokens(messages)}")
        print("[DEBUG] Sending request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        raw_content = response.choices[0].message.content.strip()
        print("[DEBUG] Raw LLM response:", raw_content)
        # Remove code block if present
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`").strip("json").strip()
        try:
            result = json.loads(raw_content)
            print(f"[DEBUG] Step 4: Detected intent: {result['intent']}, params: {result['params']}")
            print(f"[DEBUG] Count in params: {result['params'].get('count', 'not found')}")
            return result["intent"], result["params"]
        except Exception as e:
            print("OpenAI Exception:", e)
            import traceback; traceback.print_exc()
            print(f"[DEBUG] Step 4: Intent detection failed, defaulting to semantic_search")
            return "semantic_search", {"query": query}
    
    def handle_summarize_chat(self, params: dict) -> dict:
        """Handle chat summarization requests"""
        target = params.get("target")
        if not target:
            return {"error": "No target specified for chat summarization"}
        
        phone = extract_phone(target)
        if not phone:
            phone, _ = extract_contact(target)
        if not phone:
            return {"error": f"No contact found for '{target}'"}
        
        chat_msgs = [m for m in metas if m["phone"] == phone]
        if not chat_msgs:
            return {"error": f"No messages found for '{target}'"}
        
        # Get count from params, default to None (meaning whole chat)
        count = params.get("count")
        
        # Sort messages by timestamp
        chat_msgs.sort(key=lambda m: m["timestamp"])
        
        # If count is specified, take only that many messages
        if count:
            messages_to_summarize = chat_msgs[-count:]
        else:
            messages_to_summarize = chat_msgs
        
        # Check if we need to chunk the messages due to token limits
        MAX_TOKENS = 4000  # Adjust based on your model's context window
        chunk_size = 50  # Number of messages per chunk
        
        if len(messages_to_summarize) > chunk_size:
            # Split into chunks
            chunks = [messages_to_summarize[i:i + chunk_size] 
                     for i in range(0, len(messages_to_summarize), chunk_size)]
            
            summaries = []
            for chunk in chunks:
                chunk_prompt = f"""
                Summarize this portion of WhatsApp messages:
                {json.dumps([f"{m['sender']}: {m['message']}" for m in chunk], indent=2)}
                """
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user", "content": chunk_prompt}
                ]
                chunk_summary = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                ).choices[0].message.content
                summaries.append(chunk_summary)
            
            # Combine chunk summaries
            final_prompt = f"""
            Combine these summaries of different parts of the same conversation into one coherent summary:
            {json.dumps(summaries, indent=2)}
            """
            messages = [
                {"role": "system", "content": "You are a helpful assistant that combines conversation summaries."},
                {"role": "user", "content": final_prompt}
            ]
            summary = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            ).choices[0].message.content
        else:
            # For smaller conversations, summarize directly
            summary_prompt = f"""
            Summarize these WhatsApp messages:
            {json.dumps([f"{m['sender']}: {m['message']}" for m in messages_to_summarize], indent=2)}
            """
            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                {"role": "user", "content": summary_prompt}
            ]
            summary = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            ).choices[0].message.content
        
        return {
            "summary": summary,
            "messages": [f"{m['timestamp']} | {m['sender']}: {m['message']}" for m in messages_to_summarize],
            "total_messages": len(messages_to_summarize),
            "is_complete_summary": count is None  # Indicates if this is a complete chat summary
        }
    
    def handle_find_creators(self, params: dict) -> dict:
        """Handle requests to find creators who joined specific topics"""
        topics = params.get("topics", [])
        count = params.get("count", 5)
        print(f"[DEBUG] handle_find_creators - Received count: {count}")  # Add this debug print

        if not topics:
            return {"error": "No topics specified"}
            
        creators = set()
        for m in reversed(metas):
            if m["direction"] == "incoming":
                msg_lower = m["message"].lower()
                if any(topic.lower() in msg_lower for topic in topics):
                    creators.add(m["sender"])
                    if len(creators) >= count:
                        break
                        
        return {
            "creators": list(creators),
            "topics": topics,
            "count": len(creators)
        }
    
    def handle_semantic_search(self, params: dict) -> dict:
        """Handle semantic search queries"""
        query = params.get("query")
        count = params.get("count")  # Removed default count
        if not query:
            return {"error": "No search query specified"}

        # Search across ALL messages in metas
        search_limit = count * 5 if count else 1000
        results = vector_keyword_search(query, search_limit)["results"]
        
        # Process results from any chat
        unique_contacts = set()
        contact_messages = {}
        
        for r in results:
            phone = r["phone"]
            if phone not in unique_contacts:
                # Search through ALL messages in metas
                for m in metas:
                    if m["phone"] == phone and m["message"] in r["entry"]:
                        contact_messages[phone] = {
                            "timestamp": m["timestamp"],
                            "sender": m["sender"],
                            "message": m["message"]
                        }
                        unique_contacts.add(phone)
                        break
                if count and len(unique_contacts) >= count:
                    break
        
        # Format results from any chat
        formatted_results = []
        contacts_to_return = list(unique_contacts)[:count] if count else list(unique_contacts)
        for phone in contacts_to_return:
            msg_data = contact_messages[phone]
            formatted_results.append({
                "contact": phone,
                "timestamp": msg_data["timestamp"],
                "sender": msg_data["sender"],
                "message": msg_data["message"]
            })
        
        # Sort all results by timestamp
        formatted_results.sort(key=lambda x: x["timestamp"])
        
        return {
            "results": formatted_results,
            "total_contacts_found": len(unique_contacts),
            "query": query
        }
    
    def handle_keyword_search(self, params: dict) -> dict:
        """Handle exact keyword search queries"""
        keyword = params.get("keyword")
        count = params.get("count", 5)
        if not keyword:
            return {"error": "No keyword specified"}
        results = []
        for m in metas:
            if keyword.lower() in m["message"].lower():
                entry = f"{m['timestamp']} | {m['sender']}: {m['message']}"
                entry = entry.replace("\\n", "\n")
                results.append({
                    "entry": entry,
                    "phone": m["phone"]
                })
                if len(results) >= count:
                    break
        return {"results": results}
    
    def handle_analyze_creators(self, params: dict) -> dict:
        """Handle requests to analyze creator activity based on TikTok links shared"""
        timeframe = params.get("timeframe", "last_week")
        count = params.get("count", 5)
        
        # Calculate time range
        now = datetime.now(TZ)
        if timeframe == "last_week":
            start_time = now - timedelta(days=7)
        elif timeframe == "last_month":
            start_time = now - timedelta(days=30)
        else:
            return {"error": f"Unsupported timeframe: {timeframe}"}
            
        # Collect creator activity
        creator_activity = {}
        for m in metas:
            if m["direction"] == "incoming":
                # Parse timestamp
                try:
                    msg_time = datetime.strptime(m["timestamp"], "%d/%m/%Y, %H:%M")
                    msg_time = TZ.localize(msg_time)
                except:
                    continue
                    
                if msg_time < start_time:
                    continue
                    
                # Count TikTok links in message
                tiktok_links = re.findall(r'https?://(?:www\.)?tiktok\.com/[^\s]+', m["message"])
                if tiktok_links:
                    creator = m["sender"]
                    if creator not in creator_activity:
                        creator_activity[creator] = 0
                    creator_activity[creator] += len(tiktok_links)
        
        # Sort creators by activity
        sorted_creators = sorted(
            creator_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:count]
        
        # Generate summary using LLM
        summary_prompt = f"""
        Summarize the top {count} creators based on TikTok links shared in the {timeframe}:
        {json.dumps([{"creator": c, "links_shared": n} for c, n in sorted_creators], indent=2)}
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes creator activity."},
            {"role": "user", "content": summary_prompt}
        ]
        print(f"[DEBUG] Tokens for analyze_creators: {count_tokens(messages)}")
        summary = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        ).choices[0].message.content
        
        return {
            "summary": summary,
            "creators": [{"name": c, "links_shared": n} for c, n in sorted_creators],
            "timeframe": timeframe,
            "total_creators": len(creator_activity)
        }
    
    def process_query(self, query: str) -> dict:
        print(f"[DEBUG] Step 2.5: In process_query with query: {query}")
        intent, params = self.detect_intent(query)
        print(f"[DEBUG] Step 5: Dispatching to handler for intent: {intent}")
        handler = self.handlers.get(intent, self.handle_semantic_search)
        return handler(params)

# Initialize the search agent
search_agent = SearchAgent()

# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Home page with API documentation"""
    return """
    <html>
        <head>
            <title>WhatsApp Search API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
                h1 { color: #333; }
                .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
                code { background: #eee; padding: 2px 5px; border-radius: 3px; }
                pre { background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>WhatsApp Search API</h1>
            
            <div class="endpoint">
                <h2>Search Endpoint</h2>
                <p><code>POST /search</code></p>
                <p>Search through WhatsApp messages using natural language queries.</p>
                <h3>Example:</h3>
                <pre>curl -X POST http://localhost:5000/search \\
    -H "Content-Type: application/json" \\
    -d "{\\"query\\":\\"who were the top creators last week\\"}"</pre>
            </div>

            <div class="endpoint">
                <h2>Find Chat Endpoint</h2>
                <p><code>POST /find_chat</code></p>
                <p>Retrieve messages from a specific chat.</p>
                <h3>Example:</h3>
                <pre>curl -X POST http://localhost:5000/find_chat \\
    -H "Content-Type: application/json" \\
    -d "{\\"target\\":\\"+1234567890\\",\\"k\\":20}"</pre>
            </div>

            <div class="endpoint">
                <h2>Webhook Endpoint</h2>
                <p><code>POST /webhook</code></p>
                <p>Receive and process WhatsApp messages.</p>
            </div>
        </body>
    </html>
    """

@webhook_bp.route("/find_chat", methods=["POST"])
def find_chat():
    data   = request.json or {}
    target = (data.get("target") or "").strip()
    if not target:
        return jsonify(error="Target cannot be empty."),400

    phone = extract_phone(target)
    disp  = None
    if not phone:
        phone,disp = extract_contact(target)
    else:
        _,disp = extract_contact(target)

    if not phone:
        return jsonify(error=f'No contact for "{target}"'), 404

    chat_msgs = chat_map.get(phone, [])
    if not chat_msgs:
        return jsonify(error=f'No messages for "{target}"'), 404

    chat_msgs.sort(key=lambda m:m["timestamp"])
    k = int(data.get("k", 20))
    last = chat_msgs[-k:]
    return jsonify({
        "display_name": disp or phone,
        "messages": [f"{m['timestamp']} | {m['sender']}: {m['message']}" for m in last]
    })

@webhook_bp.route("/search", methods=["POST"])
def semantic_search():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    print(f"[DEBUG] Step 1: Received query: {query}")
    if not query:
        return jsonify(error="Query cannot be empty."), 400
    
    try:
        print(f"[DEBUG] Step 2: Passing query to search_agent.process_query")
        result = search_agent.process_query(query)
        print(f"[DEBUG] Step 6: Final result: {result}")
        return jsonify(result), 200
    except Exception as e:
        print("OpenAI Exception:", e)
        import traceback; traceback.print_exc()
        error_msg = str(e)
        if "insufficient_quota" in error_msg:
            return jsonify({
                "error": "OpenAI API quota exceeded. Please check billing details.",
                "details": "The current OpenAI account has exceeded its quota. Please check the billing details at https://platform.openai.com/account/billing",
                "status": "quota_exceeded"
            }), 429
        elif "model_not_found" in error_msg:
            return jsonify({
                "error": "OpenAI model not available. Please check model access.",
                "details": "The requested OpenAI model is not available. Please check your account's model access.",
                "status": "model_unavailable"
            }), 404
        else:
            print(f"[DEBUG] Search error: {error_msg}")
            print(f"[DEBUG] Error type: {type(e)}")
            print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Search failed",
                "details": error_msg,
                "status": "error"
            }), 500


@webhook_bp.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    raw = request.json.get("data",{})
    if raw.get("type")!="chat": return "ignored",200
    if raw.get("fromMe",False) and raw.get("ack")!="server":
        return "ignored",200

    process_message_event(raw)
    return "ok",200

@app.route("/contacts", methods=["GET"])
def list_contacts():
    response = [
        {"name": name.title(), "phone": phone}
        for name, phone in contacts.items()
    ]
    print(f"/contacts endpoint response: {response}")
    return jsonify(response)

# Register the webhook blueprint
app.register_blueprint(webhook_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Cloud Run's dynamic port if set
    print(f"Starting server on port {port}...")
    print(f"Visit http://localhost:{port} for API documentation")
    app.run(host="0.0.0.0", port=port, debug=False)