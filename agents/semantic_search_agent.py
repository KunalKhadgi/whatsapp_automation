#!/usr/bin/env python3
"""
Agent for semantic search using FAISS and OpenAI embedding.
All FAISS loading is done outside; index & metas must be set on the agent instance.
"""

import json
import pytz
import re
import tiktoken, datetime
import numpy as np
from datetime import timedelta
from openai import OpenAI
from utils.embed_utils import embed_texts
from utils.contact_utils import extract_contact, extract_phone

client = OpenAI()
TZ = pytz.timezone("Asia/Kolkata")


def extract_semantic_params(full_query: str) -> dict:
    """
    Ask the LLM to pull out the topic keyword and optional count
    for semantic search from the user’s query.
    """
    prompt = f"""
Extract JSON from this user request. It is asking for a semantic search.
Return only:
{{
  "query": "<the single keyword or short phrase to search>",
  "count": <optional integer>
}}

Examples:
- "who are the 20 people to whom I talked about discord" → {{"query":"discord","count":20}}
- "find people who discussed slack"          → {{"query":"slack"}}
- "show me clients who mentioned vitamin D"  → {{"query":"vitamin D"}}

Now parse:
"{full_query}"
"""
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
    )
    text = resp.choices[0].message.content.strip().strip("```json").strip("```")
    try:
        return json.loads(text)
    except Exception:
        # fallback: treat entire query as keyword
        return {"query": full_query}


class SearchAgent:
    """Agent system to handle different types of search queries"""

    def __init__(self):
        # these must be set externally:
        self.index = None
        self.metas = None
        self.contacts = None

        self.handlers = {
            "summarize_chat":     self.handle_summarize_chat,
            "find_creators":      self.handle_find_creators,
            "semantic_search":    self.handle_semantic_search,
            "keyword_search":     self.handle_keyword_search,
            "analyze_creators":   self.handle_analyze_creators
        }

    def detect_intent(self, query: str) -> tuple[str, dict]:
        """Use LLM to detect intent and extract params as JSON."""
        print(f"[DEBUG] Detecting intent for query: {query}")
        prompt = f"""
Analyze this search query and determine its intent and parameters.
Query: "{query}"

Return a JSON object with:
- intent: one of ["summarize_chat", "find_creators", "semantic_search", "keyword_search", "analyze_creators"]
- params: relevant parameters for the intent

Rules:
1. If finding people who discussed topics → semantic_search
2. If query mentions a number → count
3. For semantic search, use that topic as "query"
4. If no count, omit it

Examples:
- "who are the 20 people ... discord" → {{"intent":"semantic_search","params":{{"query":"discord","count":20}}}}
- "find people who discussed slack" → {{"intent":"semantic_search","params":{{"query":"slack"}}}}
- "summarize my chat with John" → {{"intent":"summarize_chat","params":{{"target":"John"}}}}

Return *only* the JSON.
"""
        messages = [
            {"role": "system", "content": "You are a query intent analyzer. Return only valid JSON."},
            {"role": "user",   "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "").strip()
        try:
            obj = json.loads(raw)
            return obj["intent"], obj["params"]
        except Exception as e:
            print("[WARN] intent parse failed:", e)
            return "semantic_search", {"query": query}

    def handle_summarize_chat(self, params: dict) -> dict:
        """Handle chat summarization requests"""
        target = params.get("target")
        if not target:
            return {"error": "No target specified for chat summarization"}

        # Try direct phone extraction first; otherwise fall back to contact lookup
        phone = extract_phone(target) or extract_contact(target, self.contacts)[0]
        if not phone:
            return {"error": f"No contact found for '{target}'"}

        # Now pull out all messages for that phone
        chat_msgs = [m for m in self.metas if m["phone"] == phone]
        if not chat_msgs:
            return {"error": f"No messages for '{target}'"}

        # If they asked for only the last N messages, slice; otherwise take whole chat
        count = params.get("count")
        chat_msgs.sort(key=lambda m: m["timestamp_start"])
        msgs = chat_msgs[-count:] if count else chat_msgs

        # dynamic batching under token limit
        MAX_TOKENS = 4000
        RESERVED  = 500
        enc       = tiktoken.encoding_for_model("gpt-3.5-turbo")

        batches, cur, tok = [], [], 0
        for m in msgs:
            line = f"{m['sender']}: {m['message']}"
            lt   = len(enc.encode(line))
            if tok + lt > MAX_TOKENS - RESERVED:
                batches.append(cur); cur, tok = [], 0
            cur.append(line); tok += lt
        if cur:
            batches.append(cur)

        # summarize each batch
        summaries = []
        for batch in batches:
            prompt = "Summarize these WhatsApp messages:\n" + "\n".join(batch)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user",   "content": prompt}
                ]
            )
            summaries.append(resp.choices[0].message.content)

        # combine if >1
        if len(summaries) > 1:
            final = ("Combine these summaries into one coherent summary:\n"
                     + "\n".join(f"- {s}" for s in summaries))
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content":"You are a helpful assistant that merges summaries."},
                    {"role":"user",   "content": final}
                ]
            )
            summary = resp.choices[0].message.content
        else:
            summary = summaries[0]

        return {
            "summary": summary,
            "messages": [f"{m['timestamp']} | {m['sender']}: {m['message']}" for m in msgs],
            "total_messages": len(msgs),
            "is_complete_summary": count is None
        }

    def handle_find_creators(self, params: dict) -> dict:
        """Handle 'who joined' requests (unchanged)."""
        topics = params.get("topics", [])
        count  = params.get("count", 5)
        if not topics:
            return {"error": "No topics specified"}
        seen, out = set(), []
        for m in reversed(self.metas):
            if m["direction"] == "incoming" and any(t in m["message"].lower() for t in topics):
                if m["sender"] not in seen:
                    seen.add(m["sender"])
                    out.append(m["sender"])
                    if len(out) >= count:
                        break
        return {"creators": out, "topics": topics, "count": len(out)}

    def handle_semantic_search(self, params: dict) -> dict:
        """Handle semantic search: extract keyword, vector search, return results."""
        full_q = params.get("query", "")
        if not full_q:
            return {"error": "No search query specified"}
        sem   = extract_semantic_params(full_q)
        keyword = sem.get("query")
        count   = sem.get("count", 5)

        # embed + search
        emb = embed_texts([keyword])
        dists, idxs = self.index.search(emb, count * 5)

        seen, results = set(), []
        for score, idx in zip(dists[0], idxs[0]):
            if idx < 0 or idx >= len(self.metas):
                continue
            m = self.metas[idx]
            if m["phone"] in seen:
                continue
            seen.add(m["phone"])
            results.append({
                "contact":   m["phone"],
                "timestamp": m["timestamp_start"],
                "sender":    m.get("sender_list", [None])[0],
                "message":   m["text"],
                "score":     float(score)
            })
            if len(results) >= count:
                break

        return {"query": keyword, "count": len(results), "results": results}

    def handle_keyword_search(self, params: dict) -> dict:
        """Exact substring match (unchanged)."""
        keyword = params.get("keyword")
        count   = params.get("count", 5)
        if not keyword:
            return {"error": "No keyword specified"}
        out = []
        for m in self.metas:
            if keyword.lower() in m["message"].lower():
                out.append({
                    "entry": f"{m['timestamp']} | {m['sender']}: {m['message']}",
                    "phone": m["phone"]
                })
                if len(out) >= count:
                    break
        return {"results": out}

    def handle_analyze_creators(self, params: dict) -> dict:
        """Analyze TikTok links (unchanged)."""
        timeframe = params.get("timeframe", "last_week")
        count     = params.get("count", 5)
        now = datetime.datetime.now(TZ)
        if timeframe == "last_week":
            start = now - timedelta(days=7)
        elif timeframe == "last_month":
            start = now - timedelta(days=30)
        else:
            return {"error": f"Unsupported timeframe: {timeframe}"}

        activity = {}
        for m in self.metas:
            if m["direction"] == "incoming":
                try:
                    ts = datetime.datetime.strptime(m["timestamp"], "%d/%m/%Y, %H:%M")
                    ts = TZ.localize(ts)
                except:
                    continue
                if ts < start:
                    continue
                links = re.findall(r'https?://(?:www\.)?tiktok\.com/[^\s]+', m["message"])
                if links:
                    activity[m["sender"]] = activity.get(m["sender"], 0) + len(links)

        top = sorted(activity.items(), key=lambda x: x[1], reverse=True)[:count]
        summary_prompt = f"""
Summarize the top {count} creators based on TikTok links shared in the {timeframe}:
{json.dumps([{"creator": c, "links_shared": n} for c, n in top], indent=2)}
"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes creator activity."},
                {"role": "user",   "content": summary_prompt}
            ]
        )
        summary = resp.choices[0].message.content

        return {
            "summary": summary,
            "creators": [{"name": c, "links_shared": n} for c, n in top],
            "timeframe": timeframe,
            "total_creators": len(activity)
        }

    def process_query(self, query: str) -> dict:
        """Master dispatch."""
        intent, params = self.detect_intent(query)
        handler = self.handlers.get(intent, self.handle_semantic_search)
        return handler(params)


# single instance to reuse
search_agent = SearchAgent()

# Legacy helper if needed somewhere 
def run_semantic_search(params, index, metas):
    """Legacy helper if needed elsewhere."""
    query = params.get("query")
    count = params.get("count", 5)
    emb = embed_texts([query])
    dists, idxs = index.search(emb, count * 5)
    seen, out = set(), []
    for score, i in zip(dists[0], idxs[0]):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        if m['phone'] in seen:
            continue
        seen.add(m['phone'])
        out.append({
            "score": float(score),
            "phone": m["phone"],
            "start_idx": m["start_idx"],
            "end_idx": m["end_idx"],
            "text": m["text"]
        })
        if len(out) >= count:
            break
    return {"results": out}
