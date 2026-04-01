def compute_cusum_regime(returns):
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze("columns")
    returns = pd.to_numeric(returns, errors='coerce')
    # additional processing...
