import os
import yaml
import torch
import pandas as pd
from datetime import datetime

from darts.models import NaiveMovingAverage, ARIMA, Prophet, RandomForest, BlockRNNModel, TCNModel, NBEATSModel
from darts.metrics import mape, rmse
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.data import PastCovariatesSequentialDataset


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

                # unscaled_forecast = scalers['target_scaler'].inverse_transform(forecast)
                # unscaled_test = scalers['target_scaler'].inverse_transform(test_ts)

                mape_score = mape(test_ts, forecast)
                rmse_score = rmse(test_ts, forecast)

                print(f"{model_name} - Product {idx}: MAPE = {mape_score:.2f}%  RMSE = {rmse_score:.4f}")

                forecasts.append(forecast.pd_dataframe())
                metrics.append({
                    'product_id': idx,
                    'mape': mape_score,
                    'rmse': rmse_score
                })

                forecast.pd_dataframe().to_csv(os.path.join(model_dir, f'forecast_product_{idx}.csv'))
                test_ts.pd_dataframe().to_csv(os.path.join(model_dir, f'ground_truth_product_{idx}.csv'))

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
                mape_score = mape(test_slice, forecast)
                rmse_score = rmse(test_slice, forecast)

                print(f"{model_name} - Product {idx}: MAPE = {mape_score:.2f}%  RMSE = {rmse_score:.4f}")

                forecasts.append(forecast.pd_dataframe())
                metrics.append({
                    'product_id': idx,
                    'mape': mape_score,
                    'rmse': rmse_score
                })

                forecast.pd_dataframe().to_csv(os.path.join(model_dir, f'forecast_product_{idx}.csv'))
                test_slice.pd_dataframe().to_csv(os.path.join(model_dir, f'ground_truth_product_{idx}.csv'))

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
                mape_score = mape(test_slice, forecast)
                rmse_score = rmse(test_slice, forecast)

                print(f"{model_name} - Product {idx}: MAPE = {mape_score:.2f}%  RMSE = {rmse_score:.4f}")

                forecasts.append(forecast.pd_dataframe())
                metrics.append({
                    'product_id': idx,
                    'mape': mape_score,
                    'rmse': rmse_score
                })

                forecast.pd_dataframe().to_csv(os.path.join(model_dir, f'forecast_product_{idx}.csv'))
                test_slice.pd_dataframe().to_csv(os.path.join(model_dir, f'ground_truth_product_{idx}.csv'))

        pd.DataFrame(metrics).to_csv(os.path.join(model_dir, 'metrics_summary.csv'), index=False)
