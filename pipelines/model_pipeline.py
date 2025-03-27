import os
import yaml
import torch
from darts.models import RNNModel
from darts.metrics import mape, rmse
from darts.utils.likelihood_models import GaussianLikelihood


def load_model_config(yaml_config_path):
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model_config', {})


def train_and_evaluate_rnn_model(
    train_series_scaled_list,
    test_series_scaled_list,
    full_series_scaled_list,
    full_future_covariates_scaled_list,
    output_dir,
    yaml_config_path
):
    model_config = load_model_config(yaml_config_path)
    user_rnn_config = model_config.get('rnn_model', {})
    forecasting_params = model_config.get('forecasting', {})

    base_rnn_params = {
        # Implementation of DeepAR model
        'model': 'LSTM',
        'likelihood': GaussianLikelihood(),
        'random_state': 42,
        # Compulsory parameters to provide 
        'input_chunk_length': user_rnn_config['input_chunk_length'], 
        'training_length': user_rnn_config.get('training_length'),
    }

    kwargs = {k: v for k, v in user_rnn_config.items() if k not in base_rnn_params and k != 'likelihood'}

    model = RNNModel(
        **base_rnn_params,
        **kwargs
    )

    model.fit(
        series=train_series_scaled_list,
        future_covariates=full_future_covariates_scaled_list,
        verbose=True
    )

    forecast_series_list = []
    mape_scores = []
    rmse_scores = []

    for idx, full_series_scaled in enumerate(full_series_scaled_list):
        forecast_series = model.historical_forecasts(
            series=full_series_scaled,
            future_covariates=full_future_covariates_scaled_list[idx],
            start=train_series_scaled_list[idx].end_time(),
            forecast_horizon=forecasting_params.get('forecast_horizon', 4),
            stride=forecasting_params.get('stride', 1),
            retrain=False,
            verbose=True
        )

        test_slice = full_series_scaled.slice_intersect(forecast_series)
        mape_score = mape(test_slice, forecast_series)
        rmse_score = rmse(test_slice, forecast_series)

        print(f"Product {idx}: MAPE on test (scaled): {mape_score:.2f}%")
        print(f"Product {idx}: RMSE on test (scaled): {rmse_score:.4f}")

        forecast_series_list.append(forecast_series)
        mape_scores.append(mape_score)
        rmse_scores.append(rmse_score)

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'rnn_global_model.pt'))

    return forecast_series_list, mape_scores, rmse_scores
