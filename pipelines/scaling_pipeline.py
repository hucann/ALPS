from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import Scaler, BoxCox, MissingValuesFiller, Diff


def build_scaling_pipeline(config=None):
    steps = []

    # Add scaling
    steps.append(Scaler())

    # Optionally add other time series transformations based on config
    if config:
        if config.get('boxcox_lambda'):
            steps.append(BoxCox(lmbda=config['boxcox_lambda']))

        if config.get('differencing_order'):
            steps.append(Diff(order=config['differencing_order']))

        if config.get('fill_missing', False):
            steps.append(MissingValuesFiller())

    return Pipeline(steps)


def scale_time_series_lists(train_series_list, test_series_list, future_covariates_list=None, past_covariates_list=None, config=None):
    target_scaler_pipeline = build_scaling_pipeline(config=config)

    train_series_scaled = [target_scaler_pipeline.fit_transform(ts) for ts in train_series_list]
    test_series_scaled = [target_scaler_pipeline.transform(ts) for ts in test_series_list]

    future_covariates_scaled = None
    if future_covariates_list:
        future_covariates_scaler = build_scaling_pipeline(config=config)
        future_covariates_scaled = [future_covariates_scaler.fit_transform(ts) for ts in future_covariates_list]
    else:
        future_covariates_scaler = None

    past_covariates_scaled = None
    if past_covariates_list:
        past_covariates_scaler = build_scaling_pipeline(config=config)
        past_covariates_scaled = [past_covariates_scaler.fit_transform(ts) for ts in past_covariates_list]
    else:
        past_covariates_scaler = None

    scalers = {
        "target_scaler": target_scaler_pipeline,
        "future_covariates_scaler": future_covariates_scaler,
        "past_covariates_scaler": past_covariates_scaler
    }

    return train_series_scaled, test_series_scaled, future_covariates_scaled, past_covariates_scaled, scalers
