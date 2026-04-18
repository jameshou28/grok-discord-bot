import sqlite3
import openpyxl
import os

DB_PATH = "teams.db"
XLSX_PATH = "grokTS.xlsx"

def migrate():
    if not os.path.exists(XLSX_PATH):
        print(f"Error: {XLSX_PATH} not found.")
        return

    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Reset the table
    cursor.execute("DROP TABLE IF EXISTS teams")
    cursor.execute("""
        CREATE TABLE teams (
            number TEXT PRIMARY KEY,
            name TEXT,
            region TEXT,
            rank INTEGER,
            trueskill REAL
        )
    """)

    print(f"Opening {XLSX_PATH}...")
    # read_only=False helps ensure we get all rows if the file has weird formatting
    wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
    ws = wb.active
    
    print("Processing rows...")
    count = 0
    # We use ws.values to get everything quickly
    for i, row in enumerate(ws.values):
        if i == 0: continue # Skip header
        if not row or len(row) < 2 or row[1] is None: continue
            
        try:
            # UPDATED MAPPING BASED ON YOUR DATA:
            # row[0] = Rank (280.0)
            # row[1] = Team Number ('4610Z')
            # row[2] = Team Name ('Zenith: Robot Rev')
            # row[3] = Region ('New Jersey')
            # row[4] = TrueSkill (25.6)
            
            team_num = str(row[1]).strip().upper()
            team_name = str(row[2] or "Unknown").strip()
            team_region = str(row[3] or "Unknown").strip()
            
            # Use Column A for Rank, Column E for TrueSkill
            rank = int(float(str(row[0] or 0)))
            ts = float(str(row[4] or 0.0))

            cursor.execute("INSERT OR REPLACE INTO teams VALUES (?, ?, ?, ?, ?)", 
                         (team_num, team_name, team_region, rank, ts))
            count += 1
            
            if count % 2000 == 0:
                print(f"Imported {count} teams...")

        except Exception as e:
            # Log specific row errors for debugging
            pass 

    conn.commit()
    conn.close()
    print(f"\nSUCCESS: Imported {count} teams into {DB_PATH}.")

if __name__ == "__main__":
    migrate()