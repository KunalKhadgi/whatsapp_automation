# parser.py (final clean)
import os
import re
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from filelock import FileLock
import pytz
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TZ = pytz.timezone(os.getenv('TIMEZONE', 'Asia/Kolkata'))
_DIGITS_ONLY = re.compile(r"\D+")
_DATE_ONLY = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})$")
_SKIP_PATTERNS = [
    re.compile(r"^#"),
    re.compile(r"^-{3,}"),
    re.compile(r"^Messages to this group")
]

def normalize_chat_format(txt_path: Path):
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    current_date = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if any(p.match(line) for p in _SKIP_PATTERNS):
            continue
        m_date = _DATE_ONLY.match(line)
        if m_date:
            d, mth, y = m_date.group(1).split("/")
            if len(y) == 2:
                y = "20" + y
            current_date = f"{d.zfill(2)}/{mth.zfill(2)}/{y}"
            continue
        if not current_date:
            continue
        time = "00:00"
        sender = "Unknown"
        message = line
        new_lines.append(f"{current_date}, {time} - {sender}: {message}")

    txt_path.write_text("\n".join(new_lines), encoding="utf-8")

class ZipHandler(FileSystemEventHandler):
    def on_created(self, event):
        src = Path(event.src_path)
        if not event.is_directory and src.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(src, "r") as zf:
                    for member in zf.namelist():
                        if not member.lower().endswith(".txt"):
                            continue

                        raw = Path(member).stem
                        phone = _DIGITS_ONLY.sub("", raw)
                        if not phone:
                            continue

                        dest_dir = Path(os.getenv('PROCESSED_DIR', './processed_exported_data')) / phone
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / f"{phone}.txt"

                        if dest_path.exists():
                            continue

                        with zf.open(member) as src_file, open(dest_path, "wb") as dst_file:
                            shutil.copyfileobj(src_file, dst_file)

                        normalize_chat_format(dest_path)

                archive = Path(os.getenv('BASE_DIR', '.')) / "processed_zips"
                archive.mkdir(exist_ok=True)
                shutil.move(str(src), archive / src.name)

            except Exception as e:
                print(f"Failed extracting {src}: {e}")

def bootstrap_conversation(phone: str, display_name: str):
    folder = Path(os.getenv('PROCESSED_DIR', './processed_exported_data')) / phone
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        (folder / f"{phone}.txt").touch()
        with open(folder / "meta.json", "w", encoding="utf-8") as mf:
            json.dump({"display_name": display_name}, mf)

def normalize_body(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def append_to_txt(
    phone: str,
    sender: str,
    text: str,
    timestamp: str = None
):
    """
    Append a chat line to processed_dir/phone/phone.txt, skipping
    exact-duplicate messages (ignoring minor whitespace).
    """
    folder   = Path(os.getenv('PROCESSED_DIR', './processed_exported_data')) / phone
    folder.mkdir(parents=True, exist_ok=True)
    txt_path = folder / f"{phone}.txt"

    # generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now(TZ).strftime("%d/%m/%Y, %H:%M")

    # build the full line
    line = f"{timestamp} - {sender}: {text}"

    # duplicate-skip logic: compare only the message body, normalized
    if txt_path.exists():
        content = txt_path.read_text(encoding="utf-8")
        lines   = content.splitlines()
        if lines:
            last_line = lines[-1]
            # extract the body portion after the first ": "
            parts     = last_line.split(": ", 1)
            last_body = parts[1] if len(parts) == 2 else ""
            if normalize_body(last_body) == normalize_body(text):
                # same message text (ignoring whitespace), so skip
                return

    # safe append under a file-lock
    with FileLock(str(txt_path) + ".lock"):
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    print(f"Appended to {phone}.txt: {line}")

def parse_whatsapp(txt_path: str):
    msgs = []
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*?): (.*)$')
    for ln in Path(txt_path).read_text(encoding="utf-8").splitlines():
        m = pattern.match(ln)
        if m:
            ds, ts, snd, msg = m.groups()
            try:
                dt = datetime.strptime(f"{ds} {ts}", "%d/%m/%Y %H:%M")
            except ValueError:
                continue
            msgs.append({
                "timestamp": TZ.localize(dt).isoformat(),
                "sender": snd,
                "message": msg.strip()
            })
    return msgs
