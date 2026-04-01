def compute_hmm_regime(returns, states):
    # Ensure returns is 1D
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze(axis=1)
    returns = pd.to_numeric(returns, errors='coerce').dropna()  # Convert to numeric and drop NaNs
    returns = returns.squeeze()  # Ensure returns is a Series

    data = returns.values.reshape(-1, 1)
    regime_mean = returns.groupby(states).mean()  # Group by states and calculate mean

    return data, regime_mean

# Rest of the file remains unchanged