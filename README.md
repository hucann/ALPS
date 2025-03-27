## Project Structure

```
project_root/
│
├── data/
│   └── retail_store_inventory.csv
│
├── configs/
│   └── retail_config.yaml
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
│
├── results/
│   └── forecasts/              # save forecast results and metrics for each model
│
├── main.py                     # entry point script to run full workflow (load, transform, scale, train, evaluate)
└── requirements.txt
```

---

## Workflow Summary

### 1. Data Loading & Cleaning
Use function `load_and_process_data` from `data_pipeline.py`

### 2. Data Transformation
Split into target and covariate, train and test (in Darts TimeSeries class)
Transformation
- fit_transform on training data; transform on testing data
- fit & transform for each TS

### 2. Model Training & Evaluation
Currently a single model, later build a looped model pipeline reading from a YAML or config dictionary  
Load model parameter from config file
Use `historical_forecasts()` for evaluation 

### 4. Output
Save trained models, forecasts, and evaluation metrics into:
- `models/`  
- `results/`  



