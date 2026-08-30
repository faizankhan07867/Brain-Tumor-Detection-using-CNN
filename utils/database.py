import sqlite3
import pandas as pd

from config import DATABASE_PATH


class DatabaseManager:

    def __init__(self):

        self.connection = sqlite3.connect(
            DATABASE_PATH,
            check_same_thread=False
        )

        self.cursor = self.connection.cursor()

        self.create_table()

    # ==========================================
    # Create Table
    # ==========================================

    def create_table(self):

        self.cursor.execute("""

        CREATE TABLE IF NOT EXISTS history(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_name TEXT,

            age INTEGER,

            prediction TEXT,

            confidence REAL,

            date TEXT

        )

        """)

        self.connection.commit()

    # ==========================================
    # Insert Prediction
    # ==========================================

    def insert_prediction(

        self,

        patient_name,

        age,

        prediction,

        confidence,

        date

    ):

        self.cursor.execute(

            """

            INSERT INTO history(

                patient_name,

                age,

                prediction,

                confidence,

                date

            )

            VALUES(?,?,?,?,?)

            """,

            (

                patient_name,

                age,

                prediction,

                confidence,

                date

            )

        )

        self.connection.commit()

    # ==========================================
    # Get All Records
    # ==========================================

    def get_all_predictions(self):

        return pd.read_sql_query(

            "SELECT * FROM history ORDER BY id DESC",

            self.connection

        )

    # ==========================================
    # Search Patient
    # ==========================================

    def search_patient(

        self,

        keyword

    ):

        query = """

        SELECT *

        FROM history

        WHERE patient_name LIKE ?

        ORDER BY id DESC

        """

        return pd.read_sql_query(

            query,

            self.connection,

            params=(f"%{keyword}%",)

        )

    # ==========================================
    # Delete History
    # ==========================================

    def delete_all(self):

        self.cursor.execute(

            "DELETE FROM history"

        )

        self.connection.commit()

    # ==========================================
    # Export CSV
    # ==========================================

    def export_csv(

        self,

        filename="outputs/prediction_history.csv"

    ):

        df = self.get_all_predictions()

        df.to_csv(

            filename,

            index=False

        )

        return filename

    # ==========================================
    # Close Database
    # ==========================================

    def close(self):

        self.connection.close()
