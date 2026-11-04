from fastapi import FastAPI, HTTPException
from schemas import SensorData, PhoneNumber
import requests
from database import create_tables
import sqlite3

app = FastAPI()
create_tables()

AI_SERVER = "http://127.0.0.1:8000/predict"


@app.get("/")
def home():
    return {
        "message": "Rockfall Backend Connected to AI Model!"
    }


@app.post("/predict")
def predict(data: SensorData):

    payload = {
        "alert_threshold": 0.5,
        "records": [
            data.model_dump()
        ]
    }

    try:
        response = requests.post(AI_SERVER, json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        result = response.json()

        prediction = result["predictions"][0]
        if prediction["alert"]:
            result["sms_sent"] = True
            result["sms_message"] = "Emergency alert sent to registered users."
        else:
            result["sms_sent"] = False
            result["sms_message"] = "No alert generated."

        return result

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=500,
            detail="Cannot connect to AI Server."
        )
@app.post("/register-phone")
def register_phone(data: PhoneNumber):
    conn = sqlite3.connect("rockfall.db")
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO phone_numbers(phone) VALUES (?)",
            (data.phone,)
        )
        conn.commit()

        return {
            "success": True,
            "message": "Phone number registered successfully."
        }

    except sqlite3.IntegrityError:
        return {
            "success": False,
            "message": "Phone number already exists."
        }

    finally:
        conn.close()