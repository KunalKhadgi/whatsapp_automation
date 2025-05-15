#!/usr/bin/env python3
"""
Refactored WhatsApp Processor (Shortened Version)
"""

import os, re
from flask import Flask, request, jsonify, Blueprint
from dotenv import load_dotenv
from utils.gcs_utils import download_from_gcs, download_meta_jsons_from_gcs, upload_to_gcs
from utils.contact_utils import load_contacts, extract_contact, extract_phone
from utils.embed_utils import embed_texts
from utils.parser import bootstrap_conversation
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import faiss, signal, threading
import numpy as np
from openai import OpenAI

from agents.semantic_search_agent import search_agent

# Load env
load_dotenv()
client = OpenAI()

# Flask setup
app = Flask(__name__)
webhook_bp = Blueprint("webhook", __name__)
app.register_blueprint(webhook_bp)

# Shutdown signal
shutdown_lock = threading.Lock()
shutdown_called = False

# Download vector DB at startup
BUCKET_NAME = "whatsapp-bot-2025-data"
FAISS_INDEX = download_from_gcs(BUCKET_NAME, "faiss_db/whatsapp.index", "faiss_db/whatsapp.index")
FAISS_METAS = download_from_gcs(BUCKET_NAME, "faiss_db/whatsapp_metas.npy", "faiss_db/whatsapp_metas.npy")

# Load index and metas
index = faiss.read_index(FAISS_INDEX)
metas = np.load(FAISS_METAS, allow_pickle=True).tolist()
faiss_lock = Lock()

# Contacts
download_meta_jsons_from_gcs(BUCKET_NAME, "processed_exported_data/", "processed_exported_data/")
contacts = load_contacts("processed_exported_data")

# Tell the agent about our loaded data
search_agent.index    = index
search_agent.metas    = metas
search_agent.contacts = contacts

def flush_and_upload():
    """Write local FAISS files and push them up to GCS."""
    from pathlib import Path
    import numpy as np
    
    with shutdown_lock:
        global shutdown_called
        if shutdown_called:
            return
        shutdown_called = True

    # persist locally (you may already have these variables bound)
    faiss.write_index(index, FAISS_INDEX)
    np.save(FAISS_METAS, np.array(metas, object), allow_pickle=True)

    # upload only once
    try:
        upload_to_gcs(FAISS_INDEX, BUCKET_NAME, "faiss_db/whatsapp.index")
        upload_to_gcs(FAISS_METAS, BUCKET_NAME, "faiss_db/whatsapp_metas.npy")
        print("🔄 Successfully flushed vector DB to GCS on shutdown.")
    except Exception as e:
        print("⚠️ Failed to flush vector DB to GCS:", e)

def _on_sigterm(signum, frame):
    print("SIGTERM received, flushing vector DB…")
    flush_and_upload()
    # exit immediately, or let Flask/Gunicorn wrap up
    os._exit(0)

# Register the handler
signal.signal(signal.SIGTERM, _on_sigterm)

# Endpoint: search
@app.route("/search", methods=["POST"])
def search():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify(error="Query cannot be empty."), 400

    try:
        result = search_agent.process_query(query)   # ← might already be here
        return jsonify(result), 200
    except Exception as e:
        print("[ERROR] SearchAgent:", e)
        return jsonify(error="Search failed", details=str(e)), 500

# Endpoint: find_chat
@app.route("/find_chat", methods=["POST"])
def find_chat():
    data = request.json or {}
    target = data.get("target", "")
    phone = extract_phone(target)
    if not phone:
        phone, _ = extract_contact(target, contacts)
    if not phone:
        return jsonify(error="No contact found"), 404
    chat = [m for m in metas if m["phone"] == phone]
    chat.sort(key=lambda m: m["timestamp_start"])
    return jsonify({"messages": [m["text"] for m in chat[-20:]]})

# Endpoint: webhook
@app.route("/webhook", methods=["POST"])
def webhook():
    from datetime import datetime
    raw = request.json.get("data", {})
    if raw.get("type") != "chat":
        return "ignored", 200
    fm = raw.get("fromMe", False)
    addr = raw.get("to") if fm else raw.get("from")
    chat_id = addr.replace("@c.us", "").replace("@g.us", "")
    display = raw.get("chatName") or raw.get("pushname") or chat_id
    sender = raw.get("author", "").split(":")[0].replace("@c.us", "") if "@g.us" in addr else display
    body = raw.get("body", "")
    bootstrap_conversation(chat_id, display)
    text = f"{sender}: {body}"
    vec = embed_texts([text])
    with faiss_lock:
        index.add(vec)
    metas.append({
        "uid": f"{chat_id}|live",
        "phone": chat_id,
        "text": text,
        "timestamp_start": datetime.now().isoformat(),
        "timestamp_end": datetime.now().isoformat()
    })
    faiss.write_index(index, FAISS_INDEX)
    np.save(FAISS_METAS, np.array(metas, object), allow_pickle=True)
    return "ok", 200

# Root
@app.route("/")
def home():
    return "<h1>WhatsApp Bot (Refactored)</h1><p>Use /search or /find_chat endpoints.</p>"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)