import os
import yaml
import torch
import pandas as pd
from darts.models import RNNModel, ARIMA, Prophet, RandomForest, TCNModel, NBEATSModel, NaiveMovingAverage

from darts.metrics import mape, rmse
from darts.utils.likelihood_models import GaussianLikelihood


MODEL_CLASSES = {
    'NaiveMovingAverage': NaiveMovingAverage,
    'ARIMA': ARIMA,
    'Prophet': Prophet,
    'RandomForest': RandomForest,
    'RNNModel': RNNModel,
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

    for model_name, config in model_configs.items():
        print(f"\n===== Training {model_name} =====")
        base_params = {}

        if model_name == 'NaiveMovingAverage':
            base_params = {'input_chunk_length': config.get('input_chunk_length', 12)}
        elif model_name == 'ARIMA':
            base_params = {'add_encoders': config.get('add_encoders', {})}
        elif model_name == 'Prophet':
            base_params = {
                'country_holidays': config.get('country_holidays', 'SG'),
                'add_seasonalities': config.get('add_seasonalities', []),
                'add_encoders': config.get('add_encoders', {})
            }
        elif model_name == 'RandomForest':
            base_params = {
                'lags': config.get('lags', 4),
                'lags_past_covariates': config.get('lags_past_covariates', 12),
                'lags_future_covariates': config.get('lags_future_covariates', [0, 1, 2, 3]),
                'output_chunk_length': config.get('output_chunk_length', 4),
                'add_encoders': config.get('add_encoders', {})
            }
        elif model_name == 'RNNModel':
            base_params = {
                'model': config.get('model', 'LSTM'),
                'input_chunk_length': config['input_chunk_length'],
                'training_length': config.get('training_length', 16),
                'likelihood': GaussianLikelihood() if config.get('likelihood', 'GaussianLikelihood') == 'GaussianLikelihood' else None,
                'random_state': config.get('random_state', 42),
            }
        elif model_name in ['TCNModel', 'NBEATSModel']:
            base_params = {
                'add_encoders': config.get('add_encoders', {}),
                'input_chunk_length': config.get('input_chunk_length', 12),
                'output_chunk_length': config.get('output_chunk_length', 4),
                'n_epochs': config.get('n_epochs', 20) if model_name == 'NBEATSModel' else None
            }
            base_params = {k: v for k, v in base_params.items() if v is not None}

        model_dir = os.path.join(output_dir, model_name)
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

        else:  
            model = instantiate_model(model_name, base_params, config.get('extra_args', {}))

            # Different models support different covariates
            if model_name == 'RandomForest':
                model.fit(
                    series=train_series_scaled,
                    future_covariates=future_covariates_scaled,
                    past_covariates=past_covariates_scaled
                )
            elif model_name == 'RNNModel':
                model.fit(
                    series=train_series_scaled,
                    future_covariates=future_covariates_scaled,
                    verbose=True
                )
            elif model_name in ['TCNModel', 'NBEATSModel']:
                model.fit(
                    series=train_series_scaled,
                    past_covariates=past_covariates_scaled,
                    verbose=True
                )

            for idx, full_ts in enumerate(full_series_scaled):
                forecast_args = {
                    'series': full_ts,
                    'start': train_series_scaled[idx].end_time(),
                    'forecast_horizon': 4,  # put in config 
                    'retrain': False,
                    'verbose': True, 
                    'stride': 1,  # or use default 
                }

                if model_name in ['RandomForest', 'RNNModel']:
                    forecast_args['future_covariates'] = future_covariates_scaled[idx]
                if model_name in ['RandomForest', 'TCNModel', 'NBEATSModel']:
                    forecast_args['past_covariates'] = past_covariates_scaled[idx]

                forecast = model.historical_forecasts(**forecast_args)

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
