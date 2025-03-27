from pipelines.data_pipeline import load_and_process_data
from pipelines.scaling_pipeline import scale_time_series_lists
from pipelines.model_pipeline import train_and_evaluate_rnn_model
from pipelines.save_results import save_forecast_and_metrics
import yaml
import numpy as np


def main():
    yaml_config_path = 'configs/retail_config.yaml'
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)

    file_path = config['data_file_path']

    # Load and process data with pandas pipeline
    processed_df = load_and_process_data(file_path, config)

    from darts import TimeSeries

    train_test_cfg = config['train_test_split']
    train_series_list = []
    test_series_list = []
    future_covariates_list = []

    # Convert processed df into timeseries and split
    for product_id in processed_df['Product ID'].unique():
        df_product = processed_df[processed_df['Product ID'] == product_id].set_index('Week')

        target_series = TimeSeries.from_dataframe(df_product, value_cols=[train_test_cfg['target_column']])
        future_covariates = TimeSeries.from_dataframe(df_product, value_cols=train_test_cfg['future_covariate_columns'])

        split_idx = int(len(target_series) * train_test_cfg['split_value'])
        split_point = target_series.time_index[split_idx]
        train_ts, test_ts = target_series.split_after(split_point)

        train_series_list.append(train_ts.astype(np.float32))
        test_series_list.append(test_ts.astype(np.float32))
        future_covariates_list.append(future_covariates.astype(np.float32))

    # Scale time series and covariates independently (per-series scaling)
    train_series_scaled, test_series_scaled, future_covariates_scaled, _, _ = scale_time_series_lists(
        train_series_list, test_series_list, future_covariates_list, config=None
    )

    full_series_scaled_list = [train.append(test) for train, test in zip(train_series_scaled, test_series_scaled)]

    # Train global model on scaled series and covariates without using data_transformers
    forecasts, mape_scores, rmse_scores = train_and_evaluate_rnn_model(
        train_series_scaled_list=train_series_scaled,
        test_series_scaled_list=test_series_scaled,
        full_series_scaled_list=full_series_scaled_list,
        full_future_covariates_scaled_list=future_covariates_scaled,
        output_dir='models/saved_models',
        yaml_config_path=yaml_config_path
    )

    # Save results
    save_forecast_and_metrics(
        forecast_series_list=forecasts,
        test_series_list=test_series_scaled,
        mape_scores=mape_scores,
        rmse_scores=rmse_scores,
        output_dir='results/forecasts'
    )


if __name__ == "__main__":
    main()
