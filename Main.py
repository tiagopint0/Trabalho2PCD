import datetime
import seaborn as sb
from matplotlib import pyplot as plt
from Exploration import exploration
import pandas as pd
option = 0
nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")
#Main Options
def start():
    print("\nOption 1 - Data Exploration")
    print("Option 2 - Explore a Specific Time Frame")
    print("Option 3 - Explore Pre Built Graphs")
    print("Option 4 - Ver Gráfico:")
    print("Option 5 - Ver Modelo Preditivo")
    print("Option 6 - Ver performance de diferentes modelos")
    print("Option 7 - Sair")
    
    try:
        option = int(input("Option: "))
        return option 
    except ValueError:
        print("Value can't be empty!")
        option = int(input("Option: "))
        return option

def custom_dates_graph():
    sb.lineplot(nvda, x='Date', y='Adj Close', color='b')
    ax2 = plt.twinx()
    sb.lineplot(nvda, x='Date', y='Volume', color='g', alpha=0.5, ax = ax2)
    plt.title('Closing Price (Adjusted) and Volume Traded')
    plt.ylabel('Price | Volume')
    plt.legend(['Adj Close', 'Volume'])
    plt.show()


def main():
    print("Welcome to the main analysis for the Stock Market Data from Tesla.\nSelect your option:")
    option = start()

    match option:
        case 1:
            print("Option 1")
            exploration(nvda)
        case 2:
            print("Option 2")
        case 3:
            print("Option 3")
        case 4:
            print("Option 4")
        case 5:
            print("Option 5")
        case 6:
            print("Option 6")
        case 7:
            quit()

if __name__ == '__main__':
    main()