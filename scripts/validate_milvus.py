import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from modules.storage.history_store import HistoryStore


def main():
    store = HistoryStore()
    items = store.list_recent(limit=5)
    print(f"recent_count={len(items)}")
    for it in items:
        print(it["id"], it["timestamp"], str(it["metadata"])[:80], str(it["content"])[:60])


if __name__ == "__main__":
    main()
