import os
import sqlite3

def connect():
    """Abre una conexión a la DB compartida con la app (WAL + busy_timeout, ver D7 del ROADMAP)."""
    con = sqlite3.connect(os.getenv("DATABASE_URL", "sqlite.db"), timeout=5)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA busy_timeout = 5000")
    con.execute("PRAGMA foreign_keys = ON")
    return con
