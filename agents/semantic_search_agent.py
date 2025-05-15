"""
Agent for semantic search using FAISS and OpenAI embedding.
"""
from utils.embed_utils import embed_texts
from openai import OpenAI
import json, faiss, pytz, datetime, re, tiktoken
from datetime import timedelta
import numpy as np
from ..utils.contact_utils import extract_contact, extract_phone

client = OpenAI()

TZ            = pytz.timezone("Asia/Kolkata")
DIM         = 1536
faiss_dir   = "../faiss_db"
INDEX_PATH  = faiss_dir / "whatsapp.index"
METAS_PATH  = faiss_dir / "whatsapp_metas.npy"

if INDEX_PATH.exists() and METAS_PATH.exists():
    index, metas = (
        faiss.read_index(str(INDEX_PATH), faiss.IO_FLAG_MMAP),
        np.load(str(METAS_PATH), allow_pickle=True).tolist()
    )
else:
    index, metas = faiss.IndexHNSWFlat(DIM, 32), []

# -------------- Function schemas and implementation ------------------ 

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
            {"role":"system", "content":"Return only valid JSON."},
            {"role":"user",   "content":prompt}
        ],
    )
    text = resp.choices[0].message.content.strip().strip("```json").strip("```")
    try:
        return json.loads(text)
    except Exception:
        # fallback: treat entire query as keyword
        return {"query": full_query}

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

def vector_keyword_search(keyword: str, count: int = 5) -> list[dict]:
    """
    Perform a FAISS search for the given keyword and return up to `count`
    unique contacts with their most relevant message chunk.
    """
    emb = embed_texts([keyword])
    # overfetch to dedupe by phone
    d, idxs = index.search(emb, count * 5)

    seen, out = set(), []
    for idx in idxs[0]:
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        if m["phone"] in seen:
            continue
        seen.add(m["phone"])
        # pick the *first* message in that chunk as representative
        out.append({
            "contact": m["phone"],
            "timestamp": m["timestamp_start"],
            "sender": m["sender_list"][0] if m.get("sender_list") else None,
            "message": m["text"]
        })
        if len(out) >= count:
            break
    return out

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
        
        # Dynamically pack messages into as-large-as-possible batches under token limit
        MAX_TOKENS       = 4000
        RESERVED_TOKENS  = 500   # leave headroom for system+instruction tokens
        encoder          = tiktoken.encoding_for_model("gpt-3.5-turbo")

        batches = []
        current, cur_tokens = [], 0
        for m in messages_to_summarize:
            line = f"{m['sender']}: {m['message']}"
            tok  = len(encoder.encode(line))
            # if adding this line would exceed our budget, start a new batch
            if cur_tokens + tok > (MAX_TOKENS - RESERVED_TOKENS):
                batches.append(current)
                current, cur_tokens = [], 0
            current.append(line)
            cur_tokens += tok
        if current:
            batches.append(current)

        print(f"[DEBUG] Summarizing {len(messages_to_summarize)} messages in {len(batches)} batches")

        # Summarize each batch in a single API call
        summaries = []
        for batch in batches:
            prompt = "Summarize these WhatsApp messages:\n" + "\n".join(batch)
            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                {"role": "user",   "content": prompt}
            ]
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            summaries.append(resp.choices[0].message.content)

        # If we got multiple batch summaries, combine them once
        if len(summaries) > 1:
            final_prompt = (
                "Combine these summaries into one coherent summary:\n"
                + "\n".join(f"- {s}" for s in summaries)
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant that merges summaries."},
                {"role": "user",   "content": final_prompt}
            ]
            summary = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            ).choices[0].message.content
        else:
            summary = summaries[0]
        
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
        # 1) Extract the pure keyword & count from the raw user query
        full_q = params.get("query", "")
        if not full_q:
            return {"error": "No search query specified"}
        semantic = extract_semantic_params(full_q)
        keyword = semantic.get("query")
        count   = semantic.get("count", 5)

        # 2) Perform a vector search on that keyword
        results = vector_keyword_search(keyword, count)

        # 3) Return directly
        return {
            "query":   keyword,
            "count":   len(results),
            "results": results
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



def run_semantic_search(params, index, metas):
    query = params.get("query")
    count = params.get("count", 5)

    query_vec = embed_texts([query])
    distances, indices = index.search(query_vec, count * 5)

    results = []
    seen = set()
    for score, i in zip(distances[0], indices[0]):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        if m['phone'] in seen:
            continue
        seen.add(m['phone'])
        results.append({
            "score": float(score),
            "phone": m["phone"],
            "start_idx": m["start_idx"],
            "end_idx": m["end_idx"],
            "text": m["text"]
        })
        if len(results) >= count:
            break
    return {"results": results}
