"""
database.py — SQLite Storage Engine
Handles persistent user accounts and prediction history.
"""
import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_data.db")


def init_db():
    """Initializes the database and creates tables if they don't already exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Predictions table — now includes confidence score
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT,
            diagnosis TEXT,
            class_idx INTEGER,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')

    # Add confidence column to older DBs that don't have it
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        c.execute("ALTER TABLE predictions ADD COLUMN class_idx INTEGER")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'patient'")
    except sqlite3.OperationalError:
        pass

    # Add default admin user if not exists
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'admin', 'doctor')")

    conn.commit()
    conn.close()


def create_user(username: str, password: str, role: str = 'patient') -> bool:
    """Creates a new user. Returns True if successful, False if username is taken."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    """Verifies login credentials. Returns True if valid."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return bool(row and row[0] == password)


def get_user_role(username: str) -> str:
    """Returns the role of the user (e.g. 'patient' or 'doctor')."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT role FROM users WHERE username=?", (username,))
        row = c.fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()
    if row and row[0]:
        return row[0]
    return 'patient'


def save_prediction(username: str, filename: str, diagnosis: str, class_idx: int, confidence: float):
    """Saves a prediction record to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (username, filename, diagnosis, class_idx, confidence) VALUES (?, ?, ?, ?, ?)",
        (username, filename, diagnosis, class_idx, confidence)
    )
    conn.commit()
    conn.close()


def get_history(username: str) -> pd.DataFrame:
    """
    Retrieves a user's complete prediction history from the database.
    Returns a formatted DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT filename, diagnosis, class_idx, confidence, timestamp FROM predictions "
        "WHERE username=? ORDER BY timestamp DESC",
        conn,
        params=(username,)
    )
    conn.close()

    if not df.empty:
        df.columns = ["Image Name", "Diagnosis", "Class_Idx", "Confidence", "Timestamp"]
        df["Confidence"] = df["Confidence"].apply(
            lambda x: f"{x:.1%}" if x is not None else "N/A"
        )

    return df


def get_all_patients() -> list:
    """Returns a list of all patient usernames (role='patient')."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT username FROM users WHERE role='patient' ORDER BY username")
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return [r[0] for r in rows]


def get_all_history() -> pd.DataFrame:
    """
    Retrieves complete prediction history for all users from the database.
    Returns a formatted DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT username, filename, diagnosis, confidence, timestamp FROM predictions "
        "ORDER BY timestamp DESC",
        conn
    )
    conn.close()

    if not df.empty:
        df.columns = ["Username", "Image Name", "Diagnosis", "Confidence", "Timestamp"]
        df["Confidence"] = df["Confidence"].apply(
            lambda x: f"{x:.1%}" if x is not None else "N/A"
        )

    return df
