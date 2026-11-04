from pydantic import BaseModel

class SensorData(BaseModel):
    slope_angle: float
    rock_strength: float
    rainfall_mm: float
    seismic_activity: float
    temperature_variation: float
    soil_moisture: float
    vegetation_cover: float
    mining_depth: int
    blasting_frequency: int
    weathering_index: float
    human_activity_index: float
    rainfall_intensity: float
    slope_stability_index: float
    rockfall_density: float
    rockfall_volume: float


class PredictionResponse(BaseModel):
    model_loaded: bool
    alert_threshold: float
    predictions: list
class PhoneNumber(BaseModel):
    phone: str