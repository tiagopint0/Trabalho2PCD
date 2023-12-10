import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

def exploration():
    #Reading the File
    nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")


    #Functions
    def abbrev_month(x):
        return x[:3]

    def concat_yrmonth(concat_data):
        return str(concat_data['Date'].year)+concat_data['Date'].strftime('%m')

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
    plt.show()

if __name__ == '__main__':
    exploration()