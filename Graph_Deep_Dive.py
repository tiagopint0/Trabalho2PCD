import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from ta import momentum as tam

def rsi_graph(nvda):
    nvda = nvda.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
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
    nvda = nvda.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    graph_fluctuations = nvda.loc[:,['Date','High','Low']]
    graph_fluctuations['Daily Fluctuation'] = round(graph_fluctuations['High']- graph_fluctuations['Low'],3)
    sb.lineplot(graph_fluctuations,x='Date', y='Daily Fluctuation')
    plt.title('Daily Fluctuation (High-Low)')
    plt.show()

def returns_graph(nvda):
    nvda = nvda.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    day_returns = nvda['Adj Close'].pct_change(fill_method=None).mul(100)

    sb.lineplot(x = nvda['Date'],y = day_returns.values)
    plt.title('Daily Returns (%)')
    plt.ylabel('% Return')
    plt.show()

#Este gráfico mostra nos o preço de fecho, ajustado, ao longo do tempo, com pontos em cada dia. Permite ter uma visão detalhada da evolução
#preço de mercado da NVDA ao longo dos anos.
def line_fulldate(graph_data):
    graph_data = graph_data.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    sb.lineplot(graph_data,x='Date', y='Adj Close')
    plt.title('Closing Price (Adjusted) Over Time')
    plt.show()

#Este gráfico tem um funcionamento semelhante ao de cima, mas dá nos os valores médios por ano, ao longo do tempo. Serve como base para alguma
#feature engineering, de forma a obter novos dados a partir dos existentes. É uma visão mais high level.
def line_year(graph_data):
    graph_data = graph_data.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    graph_data['Year'] = graph_data['Date'].dt.year
    graph_data = graph_data.loc[:,['Year','Adj Close']]
    graph_data = graph_data.groupby(['Year']).mean()
    sb.lineplot(graph_data,x='Year', y='Adj Close')
    plt.title('Avg. Closing Price (Adjusted) by Year')
    plt.show()

#Este gráfico é um pouco mais detalhado do que a visão anual, pois permite ver a evolução com a média mensal. Permite ver 12 pontos por ano,
#para uma visualização mais granular dos dados. Deu origem a uma função e novas features, também.
def line_month(graph_data):
    def concat_yrmonth(x):
        year_month = x.strftime('%Y-%m')
        return year_month
    
    graph_data = graph_data.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    graph_data['Year'] = graph_data['Date'].dt.year
    graph_data['Month'] = graph_data['Date'].dt.month
    graph_data['Year_Month'] = graph_data['Date'].apply(concat_yrmonth)
    graph_data = graph_data.loc[:,['Year_Month','Adj Close']]
    graph_data = graph_data.groupby(['Year_Month']).mean()
    sb.lineplot(graph_data,x='Year_Month', y='Adj Close')
    plt.title('Avg. Closing Price (Adjusted) by Year Month')
    plt.show()

#Este gráfico permite nos ver a correlação entre as diferentes variáveis, e o quanto se influenciam uma à outra. Os valores são altos, pois
#o valor de abertura, por exemplo, está altamente relacionado com o valor de fecho, por exemplo, dada a natureza dos dados.
def heatmap(nvda):
    nvda = nvda.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    heat_data = nvda.loc[:,['Open','High','Low', 'Close','Adj Close']].corr()
    sb.heatmap(heat_data)
    plt.show()

#Este gráfico permite nos ver a influência, ou não, que o volume de trading tem nos dados.
def price_over_volume_graph(nvda):
    nvda = nvda.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    sb.lineplot(nvda, x='Date', y='Adj Close')
    ax2 = plt.twinx()
    sb.lineplot(nvda, x='Date', y='Volume', alpha=0.5, ax = ax2) # type: ignore
    plt.title('Closing Price (Adjusted) and Volume Traded')
    plt.ylabel('Price | Volume')
    plt.legend(['Adj Close', 'Volume'])
    plt.show()

def display_all(nvda):
    nvda = nvda.astype({'Date': 'datetime64[ns]','Open':'float','High':'float','Low':'float','Close':'float','Adj Close':'float','Volume':'float'})
    line_fulldate(nvda)
    line_year(nvda)
    line_month(nvda)
    daily_fluc(nvda)
    price_over_volume_graph(nvda)
    returns_graph(nvda)
    heatmap(nvda)
    rsi_graph(nvda)