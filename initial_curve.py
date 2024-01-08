import os
import pandas as pd
import matplotlib.pyplot as plt

# Specify the directory where your CSV files are located
directory = "material 1"

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Loop through each CSV file and plot the data
for csv_file in csv_files:
    # Construct the full file path
    file_path = os.path.join(directory, csv_file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Select the columns of interest
    selected_cols = df[['D', 'E', 'J', 'K']]

    # Plot the data
    selected_cols.plot(title=f'Plot for {csv_file}')
    plt.show()