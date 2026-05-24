from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from location_api import get_location_features
app = FastAPI(title="Location Intelligence API")
try:
    with open("business_model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    model = None
BUSINESS_TYPES = {
    "supermart": 0,
    "grocery store": 1,
    "wine shop": 2,
    "movie theatre": 3,
    "restaurant": 4,
    "cafe": 5,
    "bakery": 6,
    "gym": 7,
    "pharmacy": 8,
    "clothing store": 9,
    "electronics shop": 10,
    "bookstore": 11,
    "salon": 12,
    "hotel": 13,
    "pet shop": 14,
    "toy store": 15,
    "mobile shop": 16,
    "car showroom": 17,
    "furniture store": 18,
    "jewelry shop": 19,
    "ice cream shop": 20,
    "sports shop": 21,
    "medical clinic": 22,
    "stationery store": 23,
    "sweet shop": 24,
    "tea stall": 25,
    "fast food outlet": 26,
    "gaming zone": 27,
    "coaching center": 28,
    "school": 29,
    "college": 30,
    "museum": 31,
    "monument": 32,
    "park": 33,
    "swimming pool": 34,
    "garden": 35,
    "custom": 36
}
class LocationRequest(BaseModel):
    lat: float
    lng: float
    business_type: str
@app.get("/")
async def home():
    return {"message": "API Running Successfully"}
@app.post("/analyze_location")
async def analyze_location(request: LocationRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not found. Run train_model.py first."
        )
    business = request.business_type.lower()
    if business not in BUSINESS_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid business type"
        )
    features = get_location_features(
        request.lat,
        request.lng
    )
    features["business_type_encoded"] = BUSINESS_TYPES[business]
    input_df = pd.DataFrame([features])
    probability = model.predict_proba(input_df)[0][1]
    return {
        "location": {
            "lat": request.lat,
            "lng": request.lng
        },
        "business_type": business,
        "success_probability": f"{round(probability * 100, 2)}%",
        "features_used": features
    }