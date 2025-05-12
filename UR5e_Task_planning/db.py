import pathlib
import sqlite3

# Ensure the database file exists in this script's directory
db_path = pathlib.Path(__file__).parent / "hand_placed.db"
db_path.touch(exist_ok=True)

# Connect and create table if not present
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS hand_placed (
        box_id INTEGER PRIMARY KEY
    )
""")
conn.commit()
conn.close()

print(f"Database initialized at {db_path}")