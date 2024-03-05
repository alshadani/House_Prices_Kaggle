import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np 
from sklearn.svm import OneClassSVM

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

def data_preprocessing(df):
    mean_non_zero = df.loc[df['MasVnrArea'] != 0, 'MasVnrArea'].mean()
    df['MasVnrArea'].fillna(mean_non_zero, inplace=True)
    df['LotFrontage'].fillna(df['LotFrontage'].mean() , inplace=True)

    # Fill in the most common points
    df['MSZoning'].fillna('RL', inplace=True)
    df['Utilities'].fillna('AllPub', inplace=True)

    df['Exterior1st'].fillna('VinylSd', inplace=True)
    df['Exterior2nd'].fillna('VinylSd', inplace=True)
    df['BsmtQual'].fillna('TA', inplace=True)
    df['BsmtCond'].fillna('TA', inplace=True)

    df['BsmtExposure'].fillna('No', inplace=True)
    df['BsmtFinType1'].fillna('Unf', inplace=True)
    df['BsmtFinType2'].fillna('Unf', inplace=True)
    df['Electrical'].fillna('SBrkr', inplace=True)

    df['GarageType'].fillna('Attchd', inplace=True)
    df['GarageYrBlt'].fillna(2005.0, inplace=True)
    df['GarageQual'].fillna('TA', inplace=True)
    df['GarageCond'].fillna('TA', inplace=True)

    # Select columns with categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Set a threshold for the maximum number of missing values allowed
    max_missing_values = 600
    # Filter columns based on the number of missing values
    selected_columns = [col for col in categorical_columns if df[col].isna().sum() <= max_missing_values]

    # Convert categorical columns to numeric using Label Encoding
    label_encoder = LabelEncoder()
    for column in selected_columns:
        df[column] = label_encoder.fit_transform(df[column].astype(str))

    columns_to_identify_outliers = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

    for column in columns_to_identify_outliers:
        # Extract the column I want to analyze
        column_to_analyze = df[column].values.reshape(-1, 1)

        # Create a One-Class SVM model
        svm_model = OneClassSVM(nu=0.3)  # You can adjust the 'nu' parameter

        # Fit the model to your data
        svm_model.fit(column_to_analyze)

        # Predict outliers
        outlier_predictions = svm_model.predict(column_to_analyze)

        # Replace outliers with the mean value
        mean_value = np.mean(df[column])
        df[column] = np.where(outlier_predictions == -1, mean_value, df[column])

        # Assuming df is your DataFrame and 'column_name' is the variable
        non_zero_values = df.loc[df['MasVnrArea'] != 0, 'MasVnrArea']
        mean_non_zero = non_zero_values.mean()
        # Replace 0 values with the mean of non-zero values
        df['MasVnrArea_no_zeros'] = np.where(df['MasVnrArea'] == 0, mean_non_zero, df['MasVnrArea'])

        # log1p normalisation for numerical features 
        df['LotFrontage_log'] = np.log1p(df['LotFrontage'])
        df['LotArea_log'] = np.log1p(df['LotArea'])
        df['MasVnrArea_log'] = np.log1p(df['MasVnrArea_no_zeros'])
        df['BsmtFinSF1_log'] = np.log1p(df['BsmtFinSF1'])
        # There is a lot of 0 values for BsmtFinSF2, it would make sense
        # in future to make from that new feuture and remove BsmtFinSF2
        df['BsmtFinSF2_log'] = np.log1p(df['BsmtFinSF2'])
        df['BsmtUnfSF_log'] = np.log1p(df['BsmtUnfSF'])
        df['TotalBsmtSF_log'] = np.log1p(df['TotalBsmtSF'])
        df['1stFlrSF_log'] = np.log1p(df['1stFlrSF'])
        # There is a lot of 0 values for 2ndFlrSF, it would make sense
        # in future to make from that new feuture and remove 2ndFlrSF
        df['2ndFlrSF_log'] = np.log1p(df['2ndFlrSF'])
        df['GrLivArea_log'] = np.log1p(df['GrLivArea'])
        df['GarageArea_log'] = np.log1p(df['GarageArea'])
        df['WoodDeckSF_log'] = np.log1p(df['WoodDeckSF'])
        df['OpenPorchSF_log'] = np.log1p(df['OpenPorchSF'])

        return df 

def feature_engineering(df):
    df['Alley_present'] = df['Alley'].notnull().astype(int)
    df['MasVnrType_present'] = df['MasVnrType'].notnull().astype(int)
    df['FireplaceQu_present'] = df['FireplaceQu'].notnull().astype(int)
    df['PoolQC_present'] = df['PoolQC'].notnull().astype(int)
    df['Fence_present'] = df['Fence'].notnull().astype(int)
    df['MiscFeature_present'] = df['MiscFeature'].notnull().astype(int)
    df['Has_garage'] = (df['GarageType'] != 6).astype(int)

    df['Gr_blt_year_combined'] = (df['GarageYrBlt'] + df['YearBuilt']) / 2
    df['Gr_blt_year_combined'] = (df['GarageYrBlt'] + df['YearBuilt']) / 2
    #df['Car_size_in_gr'] = df['GarageArea'] / df['GarageCars']

    # Columns with a lot of missing data won't be used 
    df = df.drop(columns=['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])

    return df 