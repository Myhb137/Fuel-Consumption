# 🚗 Fuel Consumption Prediction API

A machine learning project that demonstrates end-to-end data science workflow, from exploratory data analysis to production-ready API deployment. This project showcases practical application of linear regression for fuel consumption prediction based on vehicle engine specifications.

## 🎯 Project Overview

This project demonstrates a complete data science pipeline:
- **Data Analysis**: Comprehensive EDA with statistical insights and visualizations
- **Model Development**: Linear regression implementation with proper train-test split
- **API Development**: Production-ready FastAPI service with input validation
- **Deployment**: Containerized application with proper error handling

## 🔬 Technical Implementation

### Data Science Pipeline
- **Dataset**: 639 vehicle records with engine specifications and fuel consumption data
- **Features**: Engine size (liters) and cylinder count as predictive variables
- **Target**: Fuel consumption (L/100km) for regression modeling

### Model Architecture
- **Algorithm**: Linear Regression for interpretable predictions
- **Validation**: 80/20 train-test split with proper data stratification
- **Performance**: Mean Absolute Error (MAE) evaluation on test set
- **Persistence**: Model serialization using joblib for production deployment

### API Architecture
- **Framework**: FastAPI for high-performance async API development
- **Validation**: Pydantic models with input constraints and type checking
- **Error Handling**: Comprehensive exception handling with proper HTTP status codes
- 

## 📁 Project Structure

```
Fuel Consumption/
├── app.py                          # FastAPI application with production-ready error handling
├── data/
│   ├── FuelConsumption.csv         # Original dataset (639 records)
│   └── data.csv                    # Processed dataset with feature engineering
├── models/
│   └── fuel_consumption_model.pkl  # Serialized Linear Regression model
├── notebook/
│   ├── EDA.ipynb                   # Exploratory Data Analysis with statistical insights
│   └── model.ipynb                 # Model training, validation, and evaluation
├── images/
│   ├── correlation_matrix.png      # Feature correlation heatmap
│   └── engine_size_distribution.png # Engine size distribution analysis
└── venv/                          # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Fuel Consumption"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy matplotlib seaborn joblib
   ```

4. **Launch the API**
   ```bash
   uvicorn app:app --reload
   ```

5. **Access the API**
   - API Base: `http://localhost:8000`
   - Interactive Docs: `http://localhost:8000/docs`
   - Alternative Docs: `http://localhost:8000/redoc`

## 📖 API Usage

### Endpoints

#### Health Check
- **GET** `/` - API status and basic information
- **Response**: `{"message": "Fuel Consumption API is running"}`

#### Fuel Consumption Prediction
- **POST** `/predict` - Predict fuel consumption based on engine specifications
- **Input**: `{"engine_size": float, "cylinders": int}`
- **Output**: `{"prediction": float}` (L/100km)

### Usage Examples

#### Interactive Documentation
Visit `http://localhost:8000/docs` for the interactive Swagger UI where you can test all endpoints directly in your browser.

#### Python Client
```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Make prediction
url = "http://localhost:8000/predict"
payload = {
    "engine_size": 2.0,    # Engine displacement in liters
    "cylinders": 4          # Number of cylinders
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Predicted fuel consumption: {result['prediction']} L/100km")
```

#### cURL
```bash
# Health check
curl "http://localhost:8000/"

# Prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"engine_size": 2.0, "cylinders": 4}'
```

## 🔬 Model Performance & Validation

### Model Architecture
- **Algorithm**: Linear Regression for interpretable and efficient predictions
- **Feature Engineering**: Engine size and cylinder count as primary predictors
- **Data Split**: 80/20 train-test split with random state for reproducibility
- **Validation**: Mean Absolute Error (MAE) for regression evaluation

### Performance Metrics
- Model trained on 512 samples (80% of dataset)
- Tested on 127 samples (20% of dataset)
- MAE calculated on test set for unbiased performance estimation

## 🛠️ Technical Stack

### Data Science & ML
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Machine learning algorithms and model evaluation
- **Matplotlib/Seaborn**: Statistical visualizations and EDA

### Web Development
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server for production deployment
- **Joblib**: Model serialization and persistence

## 📊 Data Analysis Insights

### Dataset Characteristics
- **Size**: 639 vehicle records
- **Features**: Engine specifications and fuel consumption data
- **Quality**: Clean dataset with minimal missing values
- **Distribution**: Normal distribution of engine sizes with clear correlation patterns

### Key Findings
- Strong positive correlation between engine size and fuel consumption
- Cylinder count shows significant impact on fuel efficiency
- Linear relationship suitable for regression modeling

## 🚀 Future Enhancements

### Model Improvements
- [ ] Feature engineering with polynomial features
- [ ] Cross-validation for robust performance estimation
- [ ] Additional features (transmission type, fuel type, vehicle class)
- [ ] Model ensemble techniques for improved accuracy

### API Enhancements
- [ ] Authentication and rate limiting
- [ ] Model versioning and A/B testing
- [ ] Comprehensive logging and monitoring
- [ ] Docker containerization for deployment

### Production Readiness
- [ ] CI/CD pipeline implementation
- [ ] Automated testing suite
- [ ] Performance monitoring and alerting
- [ ] Database integration for model metadata


