def predict_risk(data):
    """
    Temporary prediction function.
    This will be replaced with the AI model later.
    """

    if data.vibration > 0.7:
        return "High", 91, 94
    else:
        return "Low", 28, 87