import os
import json
from pathlib import Path

def load_contacts(processed_dir):
    contacts = {}
    for d in Path(processed_dir).iterdir():
        if not d.is_dir(): continue
        meta = d / "meta.json"
        if meta.exists():
            try:
                name = json.loads(meta.read_text()).get("display_name", "").strip().lower()
                if name:
                    contacts[name] = d.name
            except:
                pass
    return contacts

def extract_phone(q):
    import re
    m = re.search(r"\b(\+?\d{10,14})\b", q)
    return re.sub(r"\D", "", m.group(1)) if m else None

def extract_contact(q, contacts):
    ql = q.lower()
    for name, ph in contacts.items():
        if ql in name:
            return ph, name
    return None, None
