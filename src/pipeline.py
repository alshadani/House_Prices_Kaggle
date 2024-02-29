import pandas as pd 

def show_missing_values(df):
    missing_data = df.isnull().sum()
    total_missing_data_points = missing_data.sum()

    percentage_missing = (total_missing_data_points / (df.shape[0] * df.shape[1]))

    print("Ration of missing data is:", percentage_missing)

    columns_with_missing = missing_data[missing_data > 0]

    # Display columns with missing data along with the number of missing values
    print("Columns with Missing Data:")
    for column, missing_count in columns_with_missing.items():
        print(f"{column}: {missing_count} missing values")

    print(f"\nTotal Missing Data Points: {total_missing_data_points}")
