def fetch_kospi_prices():
    # Assuming df is already defined and fetched
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close']
        if close.shape[1] == 1:
            close = close.squeeze(axis=1)
    else:
        close = df['Close']

    # Prepare the final DataFrame
    result = pd.DataFrame({
        'date': close.index,
        'kospi_price': close,
        'kospi_return': close.pct_change()
    })
    
    # Ensure date column is of datetime type
    result['date'] = pd.to_datetime(result['date'])
    
    return result
