import sqlite3

DB_NAME = "rockfall.db"

def create_tables():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS phone_numbers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone TEXT UNIQUE
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        probability REAL,
        risk INTEGER,
        alert INTEGER
    )
    """)

    conn.commit()
    conn.close()