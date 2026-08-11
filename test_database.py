from utils.database import DatabaseManager

db = DatabaseManager()

db.insert_prediction(

    "Faizan",

    22,

    "Glioma",

    98.74,

    "05-08-2026"

)

print(

    db.get_all_predictions()

)

db.close()