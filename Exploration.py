import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from ta import momentum as tam

def exploration(nvda):
    #Functions
    def concat_yrmonth(concat_data):
        return str(concat_data['Date'].year) + '-' + str(concat_data['Date'].month)
    
    def abbrev_month(x):
        return x[:3]


    #Start

    #Data Types
    print("Firstly, let's assess the data type situation.\n")
    print(nvda.dtypes)
    print("\nAs we can see, our data types are not the most optimal to work with. Let's change that. \n")
    nvda = nvda.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    print(nvda.dtypes)
    print("\nMuch better. \n")

    #NA and NULL Values

    print("Let's see if our dataset has any NA values:\n")
    na_vals = nvda[nvda.isna().any(axis=1)]
    print(na_vals)

    print("\nLet's see, now, if our dataset has any NULL values:\n")
    null_vals = nvda[nvda.isnull().any(axis=1)]
    print(null_vals)

    if len(na_vals)!=0 or len(null_vals)!=0:
        print("\nLet's delete these.")
        nvda.dropna(axis=0,inplace=True)
        print("\nNA/NULL Values:\n", nvda[nvda.isna().any(axis=1)])
        print("\nDone.")

    #Statistical Analysis
    print("\nLet's now look at some statistics about our data:\n")
    print(nvda.describe())
    print("\n\n")

    """
    Comments about the data that is being described for each column:

    Number of rows: 5000, representing 5000 different days

    Column Name: Date                Column Name: Open            Column Name: High                Column Name: Low            Column Name: Close                Column Name: Adj Close         Column Name: Volume
    Min: 1999-01-22                  Min: 1.4                     Min: 1.42                        Min: 1.33                   Min: 1.36                         Min: 1.26                      Min: 1 509 058
    Max: 2020-04-01                  Max: 312.77                  Max: 316.32                      Max: 301.49                 Max: 314.70                       Max: 314.51                    Max: 23 077 140
    Median: 2009-08-27               Median: 14.87                Median: 15.13                    Median: 14.63               Median: 14.91                     Median: 314.51                 Median: 13 308 400
    Std: NAN                         Std: 67                      Std: 68.03                       Std: 65.91                  Std: 67.01                        Std: 66.96                     Std: 1 154 203

    """

    print("\nWhat's the difference, daily, between the High, and Low values of the stock? Let's see below:\n")
    graph_fluctuations = nvda.loc[:,['Date','High','Low']]
    graph_fluctuations['Daily Fluctuation'] = round(graph_fluctuations['High']- graph_fluctuations['Low'],3)

    print(graph_fluctuations)

    #Overview Graph

    #Base Data
    graph_data = nvda.loc[:,['Date','Adj Close']]


    #Wrangling the Data for new columns
    graph_data['YearMonth'] = graph_data.apply(concat_yrmonth,axis=1)
    graph_data['Year'] = graph_data['Date'].dt.year
    graph_data['Month'] = graph_data['Date'].dt.month
    graph_data['Month Name'] = graph_data['Date'].dt.month_name().apply(abbrev_month)
    print(graph_data)

    #Graph using Seaborn
    sb.lineplot(graph_data,x='Date', y='Adj Close')
    plt.title('Closing Price (Adjusted) Over Time')
    plt.show()

    #Daily Fluctuations
    sb.lineplot(graph_fluctuations,x='Date', y='Daily Fluctuation')
    plt.title('Closing Price (Adjusted) Over Time')
    plt.show()

    #Correlation Heatmap
    heat_data = nvda.loc[:,['Open','High','Low', 'Close']].corr()
    sb.heatmap(heat_data)
    plt.show()

    #Daily Returns
    day_returns = nvda['Adj Close'].pct_change().mul(100)

    sb.lineplot(x = nvda['Date'],y = day_returns.values)
    plt.title('Daily Returns (%)')
    plt.ylabel('% Return')
    plt.show()

    #Price and Volume over Time
    sb.lineplot(nvda, x='Date', y='Adj Close', color='b')
    ax2 = plt.twinx()
    sb.lineplot(nvda, x='Date', y='Volume', color='g', alpha=0.5, ax = ax2)
    plt.title('Closing Price (Adjusted) and Volume Traded')
    plt.ylabel('Price | Volume')
    plt.legend(['Adj Close', 'Volume'])
    plt.show()

    #RSI Data
    rsi_data = nvda.loc[:,['Date','Adj Close']]
    rsi_data['rsi_monthly'] = tam.RSIIndicator(close=rsi_data['Adj Close'], window=30).rsi()

    sb.lineplot(rsi_data, x = 'Date', y = 'rsi_monthly', label='RSI (Monthly)', color='blue')
    plt.axhline(y = 30, color = 'brown', linestyle = '-', label = 'Upper Limit') 
    plt.axhline(y = 70, color = 'gray', linestyle = '-', label = 'Lower Limit') 
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.ylim([0,100])
    plt.xlim([rsi_data['Date'].min(),rsi_data['Date'].max()])
    plt.legend()
    plt.show()
if __name__ == '__main__':
    exploration(nvda=pd.DataFrame)