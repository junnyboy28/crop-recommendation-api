# Crop Recommendation API

A Flask-based REST API that provides crop recommendations based on input parameters.

## Description

This API uses a machine learning model to recommend suitable crops based on environmental parameters. The model takes into account factors such as:
- Parameter 1 (e.g., Temperature)
- Parameter 2 (e.g., Humidity)
- Parameter 3 (e.g., pH)

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/crop-recommendation-api.git
cd crop-recommendation-api
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Make a prediction:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [20, 80, 7]}'
```

### API Endpoints

- GET `/`: Health check endpoint
- POST `/predict`: Prediction endpoint

#### Prediction Request Format
```json
{
    "features": [value1, value2, value3]
}
```

#### Response Format
```json
{
    "prediction": "crop_name",
    "status": "success"
}
```

## Deployment

This API is deployed on Render. Live version can be accessed at:
[Your-Render-URL]

## Contributing

If you'd like to contribute, please fork the repository and make changes as you'd like. Pull requests are warmly welcome.

## License

[Your chosen license]

## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/yourusername/crop-recommendation-api](https://github.com/yourusername/crop-recommendation-api)