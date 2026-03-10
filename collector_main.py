from collector.database import init_db
from collector.traversal import run_traversal

if __name__ == "__main__":
    init_db()
    run_traversal()