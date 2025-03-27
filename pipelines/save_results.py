import os
import pandas as pd
from darts import TimeSeries

def save_forecast_and_metrics_per_model(model_name, forecasts, test_series, mape_scores, rmse_scores, output_dir):
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    all_results = []

    for idx, (forecast_ts, test_ts, mape_score, rmse_score) in enumerate(zip(forecasts, test_series, mape_scores, rmse_scores)):
        forecast_df = forecast_ts.pd_dataframe()
        test_df = test_ts.pd_dataframe()

        forecast_path = os.path.join(model_dir, f'forecast_product_{idx}.csv')
        ground_truth_path = os.path.join(model_dir, f'ground_truth_product_{idx}.csv')

        forecast_df.to_csv(forecast_path)
        test_df.to_csv(ground_truth_path)

        all_results.append({
            'product_id': idx,
            'mape': mape_score,
            'rmse': rmse_score,
            'forecast_file': forecast_path,
            'ground_truth_file': ground_truth_path
        })

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(os.path.join(model_dir, 'metrics_summary.csv'), index=False)
    print(f"Results saved for model: {model_name}")
    return metrics_df
