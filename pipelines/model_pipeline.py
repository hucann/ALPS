import os
import yaml
import torch
import pandas as pd
from datetime import datetime

from darts import TimeSeries
from darts.metrics import mape, rmse
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.data import PastCovariatesSequentialDataset
from darts.models import NaiveMovingAverage, ARIMA, Prophet, RandomForest, BlockRNNModel, TCNModel, NBEATSModel

MODEL_CLASSES = {
    'NaiveMovingAverage': NaiveMovingAverage,
    'ARIMA': ARIMA,
    'Prophet': Prophet,
    'RandomForest': RandomForest,
    'RNNModel': BlockRNNModel,
    'TCNModel': TCNModel,
    'NBEATSModel': NBEATSModel,
}


def load_model_config(yaml_config_path):
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model_configurations', {})


def instantiate_model(model_name, base_params, user_params):
    full_params = {**base_params, **user_params}
    if full_params.get('likelihood') == 'GaussianLikelihood':
        full_params['likelihood'] = GaussianLikelihood()
    return MODEL_CLASSES[model_name](**full_params)


def train_and_evaluate_models(
    train_series_scaled,
    test_series_scaled,
    full_series_scaled,
    future_covariates_scaled,
    past_covariates_scaled,
    scalers,
    output_dir,
    yaml_config_path
):
    model_configs = load_model_config(yaml_config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    for model_name, config in model_configs.items():
        print(f"\n===== Training {model_name} =====")
        base_params = {}

        if model_name == 'NaiveMovingAverage':
            base_params = {'input_chunk_length': config.get('input_chunk_length', 12)}
        elif model_name == 'ARIMA':
            base_params = {}
        elif model_name == 'Prophet':
            base_params = {
                'country_holidays': config.get('country_holidays', 'SG'),
                'add_seasonalities': config.get('add_seasonalities', [])
            }
        elif model_name == 'RandomForest':
            base_params = {
                'lags': config.get('lags', 4),
                'lags_past_covariates': config.get('lags_past_covariates', 12),
                'lags_future_covariates': config.get('lags_future_covariates', [0, 1, 2, 3]),
                'output_chunk_length': config.get('output_chunk_length', 4)
            }
        elif model_name == 'RNNModel':
            base_params = {
                'model': config.get('model', 'LSTM'),
                'input_chunk_length': config.get('input_chunk_length', 12),
                'output_chunk_length': config.get('output_chunk_length', 4),
                'likelihood': GaussianLikelihood() if config.get('likelihood', 'GaussianLikelihood') == 'GaussianLikelihood' else None,
            }
        elif model_name in ['TCNModel', 'NBEATSModel']:
            base_params = {
                'input_chunk_length': config.get('input_chunk_length', 12),
                'output_chunk_length': config.get('output_chunk_length', 4),
            }
            base_params = {k: v for k, v in base_params.items() if v is not None}

        model_dir = os.path.join(run_output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        forecasts = []
        metrics = []

        if model_name in ['NaiveMovingAverage','ARIMA', 'Prophet']:
            for idx, (train_ts, test_ts, full_ts, fcov_ts) in enumerate(zip(
                train_series_scaled, test_series_scaled, full_series_scaled, future_covariates_scaled)):

                model = instantiate_model(model_name, base_params, config.get('extra_args', {}))

                if model_name == 'NaiveMovingAverage':
                    model.fit(train_ts)
                    forecast = model.predict(len(test_ts))
                else:
                    model.fit(train_ts, future_covariates=fcov_ts)
                    forecast = model.predict(len(test_ts), future_covariates=fcov_ts)

                # Save unaggregated forecast and ground truth
                forecast.pd_dataframe().to_csv(os.path.join(model_dir, f'forecast_product_{idx}.csv'))
                test_ts.pd_dataframe().to_csv(os.path.join(model_dir, f'ground_truth_product_{idx}.csv'))

                # Convert to DataFrames for aggregation
                forecast_df = forecast.pd_dataframe()
                truth_df = test_ts.pd_dataframe()

                # Compute 4-week rolling sums (aligned to the DL model logic)
                forecast_df['Future_4Week_Sum'] = (
                    forecast_df['Units Sold'][::-1]
                    .rolling(window=4)
                    .sum()[::-1]
                )
                truth_df['Future_4Week_Sum'] = (
                    truth_df['Units Sold'][::-1]
                    .rolling(window=4)
                    .sum()[::-1]
                )

                # Drop last 3 values (incomplete window)
                aggregated_forecast = TimeSeries.from_dataframe(forecast_df[:-3], value_cols='Future_4Week_Sum')
                aggregated_truth = TimeSeries.from_dataframe(truth_df[:-3], value_cols='Future_4Week_Sum')

                # Evaluate
                mape_score = mape(aggregated_truth, aggregated_forecast)
                rmse_score = rmse(aggregated_truth, aggregated_forecast)

                print(f"{model_name} - Product {idx}: MAPE (4-wk sum) = {mape_score:.2f}%  RMSE = {rmse_score:.4f}")

                forecasts.append(forecast_df)
                metrics.append({
                    'product_id': idx,
                    'mape': mape_score,
                    'rmse': rmse_score
                })

        elif model_name in ['RNNModel', 'TCNModel', 'NBEATSModel']:
            input_chunk_length = config['input_chunk_length']
            output_chunk_length = config['output_chunk_length'] if model_name != 'TCNModel' else 1  # TCNModel uses 1-step forecasting

            dataset = PastCovariatesSequentialDataset(
                target_series=train_series_scaled,
                covariates=past_covariates_scaled,
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length
            )

            # === Inspect window ===
            window_past_target, window_past_covariates, _, _, window_future_target = dataset[0]  # first window
            print("\n🔍 Inspecting first training window:")
            print("\nwindow_past_target:") 
            print(window_past_target)

            print("\nwindow_past_covariates:")
            print(window_past_covariates)

            print("\nwindow_future_target:")
            print(window_future_target)
            # === Inspect window ===

            model = instantiate_model(model_name, base_params, config.get('extra_args', {}))
            model.fit_from_dataset(dataset, verbose=True)

            for idx, full_ts in enumerate(full_series_scaled):
                forecast = model.historical_forecasts(
                    series=full_ts,
                    past_covariates=past_covariates_scaled[idx],
                    start=train_series_scaled[idx].end_time(),
                    forecast_horizon= config.get('forecast_horizon', 4),
                    retrain=False,
                    verbose=True
                )

                test_slice = full_ts.slice_intersect(forecast)
                truth_df = test_slice.pd_dataframe()
                forecast_df = forecast.pd_dataframe()
                forecast_df.to_csv(os.path.join(model_dir, f'forecast_product_{idx}.csv'))
                truth_df.to_csv(os.path.join(model_dir, f'ground_truth_product_{idx}.csv'))

                # Compute future 4 weeks
                truth_df['Future_4Week_Sum'] = (
                    truth_df['Units Sold'][::-1]
                    .rolling(window=4)
                    .sum()[::-1]
                )
                truth = TimeSeries.from_dataframe(truth_df[:-3], value_cols='Future_4Week_Sum')
                forecast_df['Future_4Week_Sum'] = (
                    forecast_df['Units Sold'][::-1]
                    .rolling(window=4)
                    .sum()[::-1]
                )
                forecast = TimeSeries.from_dataframe(forecast_df[:-3], value_cols='Future_4Week_Sum')
                
                mape_score = mape(truth, forecast)
                rmse_score = rmse(truth, forecast)

                print(f"{model_name} - Product {idx}: MAPE = {mape_score:.2f}%  RMSE = {rmse_score:.4f}")

                metrics.append({
                        'product_id': idx,
                        'mape': mape_score,
                        'rmse': rmse_score
                        })
                
        else:  # RandomForest global model
            model = instantiate_model(model_name, base_params, config.get('extra_args', {}))
            model.fit(
                series=train_series_scaled,
                future_covariates=future_covariates_scaled,
                past_covariates=past_covariates_scaled
            )

            for idx, full_ts in enumerate(full_series_scaled):
                forecast = model.historical_forecasts(
                    series=full_ts,
                    future_covariates=future_covariates_scaled[idx],
                    past_covariates=past_covariates_scaled[idx],
                    start=train_series_scaled[idx].end_time(),
                    forecast_horizon= config.get('forecast_horizon', 4),
                    retrain=False,
                    verbose=True
                )

                test_slice = full_ts.slice_intersect(forecast)
                truth_df = test_slice.pd_dataframe()
                forecast_df = forecast.pd_dataframe()
                forecast_df.to_csv(os.path.join(model_dir, f'forecast_product_{idx}.csv'))
                truth_df.to_csv(os.path.join(model_dir, f'ground_truth_product_{idx}.csv'))

                # Compute future 4 weeks
                truth_df['Future_4Week_Sum'] = (
                    truth_df['Units Sold'][::-1]
                    .rolling(window=4)
                    .sum()[::-1]
                )
                truth = TimeSeries.from_dataframe(truth_df[:-3], value_cols='Future_4Week_Sum')
                forecast_df['Future_4Week_Sum'] = (
                    forecast_df['Units Sold'][::-1]
                    .rolling(window=4)
                    .sum()[::-1]
                )
                forecast = TimeSeries.from_dataframe(forecast_df[:-3], value_cols='Future_4Week_Sum')
                
                mape_score = mape(truth, forecast)
                rmse_score = rmse(truth, forecast)

                print(f"{model_name} - Product {idx}: MAPE = {mape_score:.2f}%  RMSE = {rmse_score:.4f}")

                metrics.append({
                        'product_id': idx,
                        'mape': mape_score,
                        'rmse': rmse_score
                        })

        pd.DataFrame(metrics).to_csv(os.path.join(model_dir, 'metrics_summary.csv'), index=False)
