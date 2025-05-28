## Overview

This project forecasts SKU-level inventory using multiple time series forecasting models via the Darts library. It includes data preprocessing, feature engineering, model training, and result visualization – all configurable via YAML and modular pipelines.

---
### How to Run
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset/data) and place it under the `data/` folder as `retail_store_inventory.csv`.
2. (Optional but recommended) Create a virtual environment (Python 3.12) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the full pipeline with:
   ```bash
   python main.py
   ```
   
### How to Modify
- To customize preprocessing steps (e.g., drop columns, apply filters), modify `pipelines/data_pipeline.py`.
- To configure or switch models, tune hyperparameters, or change training parameters, update `configs/retail_config.yaml`.


---
### Project Structure
```
project_root/
│
├── data/                       # Raw dataset (place downloaded CSV here) 
│
├── configs/                    # YAML config for models and training  
│
├── pipelines/                  # Core pipeline scripts
│   ├── data_pipeline.py        # Load, clean, and transform raw data
│   ├── scaling_pipeline.py     # Apply scaling and train-test split
│   ├── model_pipeline.py       # Define/train models, forecast, evaluate
│   └── save_results.py         # Save forecasts and metrics
│
├── notebooks/                  # Exploratory analysis and experiments
│   └── exploration.ipynb       # general exploration
│   └── visualization_forecast.ipynb   # visualize forecast result and metric 
│   └── regression.ipynb        # explore aggregated target VS multiple single target 
│   └── aggregate.ipynb         # explore aggregated value as target value
│   └── tsfel_feature_engi.ipynb       # explore automated feature engineering by TSFEL
│   └── multivariate.ipynb      # explore multivaraite VS global model
│
├── results/                    # Output directory for forecasts/metrics
│   └── forecasts/              
│
├── main.py                     # Entry script to run full pipeline
│
└── requirements.txt            # Project dependencies
```

---
### Data & Model Pipeline

#### 1. Data Loading & Cleaning
- Implemented in `pipelines/data_pipeline.py` → function `load_and_process_data()`
- Handles reading, filtering, missing values, aggregation, and encoding.

#### 2. Data Transformation & Splitting
- Converts data to Darts `TimeSeries` objects.
- Splits into:
  - Target series
  - Past covariates (e.g. promotions, holidays)
  - Train/test sets (time-based split)
- Handled in `scaling_pipeline.py`

#### 3. Model Training & Evaluation
- Models configured in `configs/retail_config.yaml`
- Supported models: 
  - Statistical: Naive, ARIMA, Prophet
  - ML/DL: RandomForest, RNN (or DeepAR), NBEATS, TCN
- Uses Darts `historical_forecasts()` for rolling evaluation.

#### 4. Output
- Forecasts and evaluation metrics saved to `results/forecasts/` as `.csv` 


---
### Key Concepts from Darts Library
- **Covariates**: exogenous variables that can improve forecasting. See [Covariates in Darts](https://unit8co.github.io/darts/userguide/covariates.html).
- **Global vs Local Models**: global models learn patterns across series. [Read more](https://unit8.com/resources/training-forecasting-models)
- **Windowing**: controlled by: 
  - `input_chunk_length` – lookback/context window
  - `output_chunk_length` – forecast/prediction horizon
  - To customize slicing, use [SequentialDataset](https://unit8co.github.io/darts/generated_api/darts.utils.data.sequential_dataset.html) and `fit_from_dataset()` instead of `fit()`. See implementation examples in `notebooks/aggregate.ipynb`.

