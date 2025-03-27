import os
import pandas as pd
from darts import TimeSeries

def save_forecast_and_metrics(forecast_series_list, test_series_list, mape_scores, rmse_scores, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for idx, (forecast_series, test_series, mape_score, rmse_score) in enumerate(zip(forecast_series_list, test_series_list, mape_scores, rmse_scores)):
        forecast_df = forecast_series.pd_dataframe()
        test_df = test_series.pd_dataframe()

        # Save forecasts
        forecast_path = os.path.join(output_dir, f'forecast_product_{idx}.csv')
        forecast_df.to_csv(forecast_path)

        # Save corresponding ground truth
        ground_truth_path = os.path.join(output_dir, f'ground_truth_product_{idx}.csv')
        test_df.to_csv(ground_truth_path)

        # Save metrics to dictionary
        all_results.append({
            'product_id': idx,
            'mape': mape_score,
            'rmse': rmse_score,
            'forecast_file': forecast_path,
            'ground_truth_file': ground_truth_path
        })

    # Save metrics summary
    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    print(f"Forecasts, ground truths, and metrics saved to {output_dir}")
