import datetime
import Exploration
option = 0

#Main Options
def start():
    print("\nOption 1 - Data Exploration")
    print("Option 2 - Explore Pre Built Graphs")
    print("Option 3 - Explore Custom Parameters for Graphs")
    print("Option 4 - Exit")
    try:
        option = int(input("Option:"))
        return option 
    except ValueError:
        print("Value can't be empty!")
        option = int(input("Option:"))
        return option

print("Welcome to the main analysis for the Stock Market Data from Tesla.\nSelect your option:")
option = start()

match option:
    case 1:
        print("Option 1")
    case 2:
        print("Option 2")
    case 3:
        print("Option 3")
    case 4:
        print("Option 4")