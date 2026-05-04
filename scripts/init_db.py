import sqlite3
from pathlib import Path

DB_PATH = Path("data/demo.db")

def main():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS products")

    cursor.execute(
        """
        CREATE TABLE products(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL)
    """
    )

    products = [
        ("Keyboard","electronics",199.0,30),
        ("Mouse","electronics",99.0,50),
        ("Monitor","electronics",1299.0,12),
        ("Notebook","office",15.0,200),
        ("Pen","office",3.5,500),
        ("Backpack","daily",159.0,20),
    ]

    cursor.executemany(
        """
        INSERT INTO products(name,category,price,stock)
        VALUES(?,?,?,?)
        """,
        products,
    )

    conn.commit()
    conn.close()

    print(f"Database initialized at {DB_PATH}")


if __name__=="__main__":
    main()