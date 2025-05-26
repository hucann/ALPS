## how to run 
1. Download [dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset/data) and place under folder data/
2. Run main.py

## how to modify
- Edit `data_pipeline.py` to preprocess data given a dataset 
- Edit `retail_config.yaml` to add additional parameter for model definition or training

## Project Structure

```
project_root/
│
├── data/
│   └── retail_store_inventory.csv
│
├── configs/
│   └── retail_config.yaml      # contains model definition and training paramters 
│
├── pipelines/
│   ├── data_pipeline.py        # contains data loading, dropping columns, aggregation, encoding
│   ├── scaling_pipeline.py     # contains scaling (after converting to TimeSeries and splits train-test)
│   ├── model_pipeline.py       # contains model definition, model training, forecasting, evaluation
│   └── save_results.py         # contains saving of results and evaluation metrics
│
├── models/
│   └── saved_models/           # for saving model checkpoints
│
├── notebooks/
│   └── exploration.ipynb       # optional for EDA or visualization
│   └── visualization_forecast.ipynb       # visualize forecast result and metric 
│   └── regression.ipynb        # explore aggregated target VS multiple single target 
│   └── aggregate.ipynb         # explore aggregated value as target value
│   └── tsfel_feature_engi.ipynb       # explore automated feature engineering by TSFEL
│   └── multivariate.ipynb      # explore multivaraite VS global model
├── results/
│   └── forecasts/              # save forecast results and metrics for each model for each experiment
│
├── main.py                     # entry point script to run full workflow (load, transform, scale, train, evaluate)
└── requirements.txt
```


## Workflow Summary

### 1. Data Loading & Cleaning
Use function `load_and_process_data` from `data_pipeline.py`

### 2. Data Transformation
Split into target and covariate, train and test (in Darts TimeSeries class)
Transformation

### 3. Model Training & Evaluation
Multiple forecasting models (including NaiveMovingAverage, ARIMA, Prophet, RandomForest, TCN, NBEATS, RNN) configured via YAML config file
Load model parameter from config file
Use `historical_forecasts()` for evaluation 

### 4. Output
Save trained models, forecasts, and evaluation metrics into:
- `models/`  
- `results/`  



