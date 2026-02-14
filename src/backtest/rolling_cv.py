import pandas as pd

def rolling_splits(dates: pd.Series, train_days: int, horizon_days: int, step_days: int):
    """
    Génère des (train_end, test_start, test_end) sur une série de dates triées.
    """
    uniq = pd.Series(pd.to_datetime(dates.unique())).sort_values().reset_index(drop=True)
    if len(uniq) < train_days + horizon_days:
        return

    start_idx = train_days - 1
    while start_idx + horizon_days < len(uniq):
        train_end = uniq.iloc[start_idx]
        test_start = uniq.iloc[start_idx + 1]
        test_end = uniq.iloc[start_idx + horizon_days]
        yield train_end, test_start, test_end
        start_idx += step_days