import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from ta import momentum as tam

def rsi_graph(nvda):
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

def daily_fluc(nvda):
    graph_fluctuations = nvda.loc[:,['Date','High','Low']]
    graph_fluctuations['Daily Fluctuation'] = round(graph_fluctuations['High']- graph_fluctuations['Low'],3)
    sb.lineplot(graph_fluctuations,x='Date', y='Daily Fluctuation')
    plt.title('Closing Price (Adjusted) Over Time')
    plt.show()

def returns_graph(nvda):
    day_returns = nvda['Adj Close'].pct_change().mul(100)

    sb.lineplot(x = nvda['Date'],y = day_returns.values)
    plt.title('Daily Returns (%)')
    plt.ylabel('% Return')
    plt.show()

def line_fulldate(graph_data):
    sb.lineplot(graph_data,x='Date', y='Adj Close')
    plt.title('Closing Price (Adjusted) Over Time')
    plt.show()

def line_year(graph_data):
    graph_data['Year'] = graph_data['Date'].dt.year
    graph_data = graph_data.loc['Year','Adj Close']
    graph_data = graph_data.group_by(['Year']).mean()
    sb.lineplot(graph_data,x='Year', y='Adj Close')
    plt.title('Avg. Closing Price (Adjusted) by Year')
    plt.show()

def line_month(graph_data):
    def concat_yrmonth(concat_data):
        return str(concat_data['Date'].year) + '-' + str(concat_data['Date'].month)
    
    graph_data['Year'] = graph_data['Date'].dt.year
    graph_data['Month'] = graph_data['Date'].dt.month
    graph_data['Year_Month'] = graph_data['Date'].apply(concat_yrmonth)

    graph_data = graph_data.loc['Year_Month','Adj Close']
    graph_data = graph_data.group_by(['Year_Month']).mean()
    sb.lineplot(graph_data,x='Year_Month', y='Adj Close')
    plt.title('Avg. Closing Price (Adjusted) by Year Month')
    plt.show()


def heatmap(nvda):
    heat_data = nvda.loc[:,['Open','High','Low', 'Close']].corr()
    sb.heatmap(heat_data)
    plt.show()

def price_over_volume_graph(nvda):
    sb.lineplot(nvda, x='Date', y='Adj Close', color='b')
    ax2 = plt.twinx()
    sb.lineplot(nvda, x='Date', y='Volume', color='g', alpha=0.5, ax = ax2)
    plt.title('Closing Price (Adjusted) and Volume Traded')
    plt.ylabel('Price | Volume')
    plt.legend(['Adj Close', 'Volume'])
    plt.show()

def display_all(nvda):
    line_fulldate(nvda)
    line_year(nvda)
    line_month(nvda)
    daily_fluc(nvda)
    price_over_volume_graph(nvda)
    returns_graph(nvda)
    heatmap(nvda)
    rsi_graph(nvda)