import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, data):
        self.data = data

    def calculate_returns(self):
        self.data['returns'] = self.data['close'].pct_change()
        avg_ret = self.data['returns'].mean()
        val_rate = (self.data['returns'] > 0).mean() * 100
        print(f"Average return: {avg_ret:+.2f}%")
        print(f"Valuation rate: {val_rate:.0f}%")

    def plot_anomalies(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.data['date'], self.data['close'])
        plt.title('Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()  

if __name__ == '__main__':
    df = pd.read_csv('your_data.csv')  # Change to your actual data file
    detector = AnomalyDetector(df)
    detector.calculate_returns()
    detector.plot_anomalies()