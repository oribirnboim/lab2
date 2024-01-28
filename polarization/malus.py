import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data(filename):
    try:
        # Assuming the Excel file is in the same directory as the script
        path = filename if filename.endswith('.xlsx') else filename + '.xlsx'
        
        # Reading only the first and second columns of the Excel file into a DataFrame
        df = pd.read_excel(path, usecols=[0, 1], skiprows=6)
        
        # Converting the DataFrame columns to NumPy arrays
        array1 = df.iloc[:, 0].to_numpy()
        array2 = df.iloc[:, 1].to_numpy()
        
        # Returning the NumPy arrays
        return array1, array2
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def process_data(data):
    return data



def malus2():
    data = get_data('malus2')
    data = process_data(data)
    x, y = data
    plt.plot(x, y)
    plt.grid()
    plt.show()



if __name__ == "__main__":
    malus2()
