import pandas as pd

class DataPipeline:
    def __init__(self, config):
        self.config = config

    def drop_columns(self, df):
        if self.config.get('columns_to_drop'):
            df = df.drop(columns=self.config['columns_to_drop'])
        return df

    def handle_missing_values(self, df):
        strategy = self.config.get('missing_value_handling', 'interpolate')
        if strategy == 'interpolate':
            return df.interpolate()
        elif strategy == 'ffill':
            return df.fillna(method='ffill')
        elif strategy == 'bfill':
            return df.fillna(method='bfill')
        return df

    def aggregate_data(self, df, aggregation_cfg):
        def custom_mode(series):
            return series.mode()[0] if not series.mode().empty else None

        agg_dict_resolved = {}
        for key, value in aggregation_cfg['agg_dict'].items():
            if isinstance(value, dict) and value.get('function') == 'mode':
                agg_dict_resolved[key] = custom_mode
            else:
                agg_dict_resolved[key] = value

        return df.groupby(aggregation_cfg['groupby_cols']).agg(agg_dict_resolved).reset_index()

    def add_week_column(self, df):
        df['Week'] = df['Date'].dt.to_period('W').astype(str)
        return df

    def reformat_week_column(self, df):
        df['Week'] = pd.to_datetime(df['Week'].str.split('/').str[0])
        df = df.sort_values(by=['Product ID', 'Week'])
        return df

    def encode_columns(self, df):
        if self.config.get('encoding_columns'):
            for col in self.config['encoding_columns']:
                df[col] = df[col].astype('category').cat.codes
        return df

    def process(self, df):
        df = self.drop_columns(df)
        df = self.handle_missing_values(df)

        if self.config.get('aggregate_daily'):
            df = self.aggregate_data(df, self.config['aggregate_daily'])

        if self.config.get('add_week_column'):
            df = self.add_week_column(df)

        if self.config.get('aggregate_weekly'):
            df = self.aggregate_data(df, self.config['aggregate_weekly'])

        if self.config.get('reformat_week_column'):
            df = self.reformat_week_column(df)

        df = self.encode_columns(df)

        return df


def load_and_process_data(file_path, config):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    pipeline = DataPipeline(config)
    return pipeline.process(df)
