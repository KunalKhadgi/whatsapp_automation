

"""
Keyword-based search over metadata entries.
"""
def run_keyword_search(params, metas):
    keyword = params.get("keyword", "").lower()
    count = params.get("count", 5)
    results = []
    for m in metas:
        if keyword in m["text"].lower():
            results.append({
                "phone": m["phone"],
                "text": m["text"],
                "timestamp": m["timestamp_start"]
            })
            if len(results) >= count:
                break
    return {"results": results}
