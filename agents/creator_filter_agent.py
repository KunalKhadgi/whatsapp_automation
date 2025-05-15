"""
Agent that filters creators who confirmed joining specified topics.
"""
def run_creator_filter(params, metas):
    topics = params.get("topics", [])
    count = params.get("count", 5)
    seen, result = set(), []

    for m in reversed(metas):
        if m.get("direction_list") and "incoming" not in m["direction_list"]:
            continue
        if any(topic.lower() in m["text"].lower() for topic in topics):
            if m["phone"] not in seen:
                seen.add(m["phone"])
                result.append({
                    "phone": m["phone"],
                    "text": m["text"],
                    "matched_topics": [t for t in topics if t in m["text"].lower()]
                })
        if len(result) >= count:
            break
    return {"creators": result}
