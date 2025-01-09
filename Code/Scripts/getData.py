import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# We choose 100 US Corporations from the S&P 500 index (from Stock Analysis website)
sp500_tickers = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "GOOG", "META", "TSLA", "AVGO", "DELL",
    "WMT", "LLY", "JPM", "V", "XOM", "MA", "UNH", "ORCL", "COST", "HD",
    "PG", "NFLX", "BAC", "JNJ", "CRM", "ABBV", "CVX", "KO", "MRK", "TMUS",
    "WFC", "CSCO", "NOW", "ACN", "BX", "AXP", "MCD", "AMD", "MS", "TMO",
    "IBM", "PEP", "DIS", "LIN", "ABT", "GS", "ISRG", "ADBE", "PM", "GE",
    "QCOM", "BWA", "CAT", "TXN", "INTU", "DHR", "VZ", "T", "BKNG", "BLK",
    "SPGI", "RTX", "PFE", "HAS", "HON", "AMAT", "NEE", "CMCSA", "PGR", "LOW",
    "PYPL", "AMGN", "UNP", "ETN", "SCHW", "C", "SYK", "KKR", "TJX", "BSX",
    "COP", "BA", "PANW", "FI", "ADP", "DE", "BMY", "GILD", "LMT", "MU",
    "ADI", "CB", "NKE", "SBUX", "UPS", "MDT", "VRTX", "MMC", "LRCX", "INTC"
]

# Define the time period - in the paper they used 2008-01-01 to 2013-02-28, but we will use a more recent period in order to minimize missing data
start_date = "2019-01-01"
end_date = "2024-02-28"

def get_equity_data(save = False):
    equity_data = {}

    for ticker in sp500_tickers:
        try:
            # Download historical data from Yahoo Finance
            data = yf.download(ticker, start=start_date, end=end_date)
            equity_data[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    daily_equity_returns = pd.DataFrame()

    for df in equity_data.values():
        df['DailyReturn'] = df['Close'].pct_change() * 100
        daily_equity_returns = pd.concat([daily_equity_returns, df['DailyReturn']], axis=1)

    daily_equity_returns.columns = equity_data.keys()

    print("--- Data fetched successfully ---")
    print("Shape of the data (T, m):", daily_equity_returns.shape)
    print("Number of missing values:", daily_equity_returns.isnull().sum().sum())
    print("missing values:", daily_equity_returns.isnull().sum())

    # Save the data to a CSV file
    if save:
        daily_equity_returns.to_csv("../Data/daily_equity_returns_recent.csv")


#For the CDS data, we would need a financial data provider like Bloomberg or Markit and not Yahoo Finance.
def get_CDS(save = False):
    print("Fetching CDS data...")






if __name__ == "__main__":
    get_equity_data(save=False)
    get_CDS(save=False)