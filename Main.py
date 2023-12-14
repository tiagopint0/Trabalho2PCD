import datetime
import seaborn as sb
from matplotlib import pyplot as plt
from Exploration import exploration
from LSTM_Implement import lstm_model
from LR_Implement import lr_model
from Graph_Deep_Dive import *

import pandas as pd
option = 0
nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")
nvda['Date'] = nvda['Date'].astype('datetime64[ns]')
#Main Options
def start():
    print("\nOption 1 - Data Exploration")
    print("Option 2 - Explore a Specific Time Frame")
    print("Option 3 - Explore Pre Built Graphs")
    print("Option 4 - Graphs:")
    print("Option 5 - View Prediction Model")
    print("Option 6 - View Different Model Performance")
    print("Option 7 - Exit")
    
    try:
        option = int(input("Option: "))
        return option 
    except ValueError:
        print("Invalid Option.")
        option = int(input("Option: "))
        return option


def custom_dates_graph():
    start_date = (input("Start Date (yyyy-mm-dd): "))
    end_date = (input("End Date (yyyy-mm-dd): "))
    nvda_custom = nvda[(start_date<=nvda['Date']) & (nvda['Date']<=end_date)]
    sb.lineplot(nvda_custom,x=nvda_custom['Date'], y='Adj Close')
    plt.title('Closing Price (Adjusted) Over Time')
    plt.ylabel('Price')
    plt.show()


def main():
    print("Welcome to the main analysis for the Stock Market Data from Tesla.\nSelect your option:")
    option = start()

    match option:
        case 1:
            exploration(nvda)
        case 2:
            custom_dates_graph()
        case 3:
            display_all(nvda)
        case 4:
            print("1 - Closing Price Over Time")
            print("2 - Average Closing Price by Year")
            print("3 - Average Closing Price by Year-Month")
            print("4 - Daily Fluctuation")
            print("5 - Correlation Heatmap")
            print("6 - Daily Returns (Percentages)")
            print("7 - Closing Price | Volume Over Time")
            print("8 - RSI Graph")
            print("9 - Return to Previous Menu")
            try:
                option_4 = int(input("Option: "))
            except ValueError:
                print("Invalid Option.")
                option_4 = int(input("Option: "))
            match option_4:
                case 1:
                    line_fulldate(nvda)
                case 2:
                    line_year(nvda)
                case 3:
                    line_month(nvda)
                case 4:
                    daily_fluc(nvda)
                case 5:
                    heatmap(nvda)
                case 6:
                    returns_graph(nvda)
                case 7:
                    price_over_volume_graph(nvda)
                case 8:
                    rsi_graph(nvda)
                case 9:
                     main()      
        case 5:
            lstm_model(nvda)
        case 6:
            print("1 - LSTM Model")
            print("2 - LR Model")
            print("3 - Return to Previous Menu")
            try:
                option_6 = int(input("Option: "))
            except ValueError:
                print("Invalid Option.")
                option_6 = int(input("Option: "))
            match option_6:
                case 1:
                    lstm_model(nvda)
                case 2:
                    lr_model(nvda)
                case 3:
                    main()
        case 7:
            quit()

if __name__ == '__main__':
    main()