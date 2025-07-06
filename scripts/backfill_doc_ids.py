import hashlib
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vecpipe.config import settings

DB_PATH = str(settings.WEBUI_DB)


def backfill_doc_ids():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id, path FROM files WHERE doc_id IS NULL")
    files_to_update = c.fetchall()

    for file_id, path in files_to_update:
        doc_id = hashlib.md5(path.encode()).hexdigest()[:16]
        c.execute("UPDATE files SET doc_id = ? WHERE id = ?", (doc_id, file_id))

    conn.commit()
    conn.close()

    print(f"Updated {len(files_to_update)} files with doc_ids.")


if __name__ == "__main__":
    backfill_doc_ids()
