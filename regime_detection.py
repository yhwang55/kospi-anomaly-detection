def compute_hmm_regime(data, model):
    model.fit(data)
    states = model.predict(data)
    return states
