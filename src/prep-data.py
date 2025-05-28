
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from azure.identity import DefaultAzureCredential
import os
import joblib

def main(args):
    input_path = args.input_data
    output_dir = args.output_data
    scaler_full_output_path = args.scaler_output_data
    num_cols = ['Rotational speed', 'Tool wear']
    float_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    threshold_unique = 90
    infrequent_threshold = 0.05

    print(f"Reading data from: {input_path}")
    df = read_data(input_path)
    print(f"Initial data shape: {df.shape}")

    print("Cleaning data...")
    cleaned_df = clean_data(df)
    print(f"Cleaned data shape: {cleaned_df.shape}")

    print("Renaming features...")
    renamed_df = rename_features(cleaned_df)
    print(f"Shape after renaming: {renamed_df.shape}")

    print("Converting numerical columns to float...")
    normalized_df = num_to_float(renamed_df, num_cols)
    print(f"Shape after float conversion: {normalized_df.shape}")

    print(f"Dropping columns with more than {threshold_unique}% unique values...")
    dropped_df = drop_high_unique_cols(normalized_df, threshold=threshold_unique)
    print(f"Shape after dropping high unique columns: {dropped_df.shape}")

    print("Creating 'Failure type' column...")
    add_col_df = create_failure_type(dropped_df)
    print(f"Shape after creating 'Failure type': {add_col_df.shape}")

    print("Encoding categorical features...")
    encoded_df = label_enc(add_col_df)
    print(f"Shape after encoding: {encoded_df.shape}")

    print("Scaling numerical features...")
    scaled_df, scaler = scaling_data(encoded_df, float_cols)
    print(f"Shape after scaling: {scaled_df.shape}")

    print("Removing target anomalies...")
    removed_anom = target_anomaly(scaled_df)
    print(f"Shape after removing target anomalies: {removed_anom.shape}")

    print(f"Dropping infrequent 'Failure type' (<= {infrequent_threshold*100}%)...")
    cleaned_failure = drop_infrequent(removed_anom, threshold=infrequent_threshold)
    print(f"Shape after dropping infrequent failures: {cleaned_failure.shape}")

    print("Augmenting data using SMOTENC...")
    augmented_data = augment_data(cleaned_failure)
    print(f"Shape after augmentation: {augmented_data.shape}")

    parent_dir = os.path.dirname(output_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Parent output directory ensured: {parent_dir}")
    else:
        print(f"Warning: Output path '{output_dir}' has no parent directory. Writing to current directory.")
    print(f"Writing prepared data to: {output_dir}")
    augmented_data.to_csv(output_dir, index=False)
    
    row_count = (len(augmented_data))
    print(f'Prepared {row_count} rows of data')
    print(augmented_data.info())

    os.makedirs(os.path.dirname(scaler_full_output_path), exist_ok=True)
    joblib.dump(scaler, scaler_full_output_path)
    print(f"StandardScaler saved to: {scaler_full_output_path}")

def read_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def rename_features(df):
    df.rename(mapper={'Type': 'Machine type', 'Air temperature [K]': 'Air temperature',
                      'Process temperature [K]': 'Process temperature',
                      'Rotational speed [rpm]': 'Rotational speed',
                      'Torque [Nm]': 'Torque',
                      'Tool wear [min]': 'Tool wear'}, axis=1, inplace=True)
    return df

def num_to_float(df, num_cols):
    df[num_cols] = df[num_cols].astype(float)
    return df

def drop_high_unique_cols(df, threshold):
    init_cols = df.columns.tolist()
    num_rows = len(df)
    drop_cols = []
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_percentage = unique_count / num_rows * 100
        if unique_percentage > threshold:
            drop_cols.append(col)
    df = df.drop(columns=drop_cols)
    return df

def create_failure_type(df):
    df['Failure type'] = 'No Failure'
    df.loc[df['TWF'] == 1, 'Failure type'] = 'TWF'
    df.loc[df['HDF'] == 1, 'Failure type'] = 'HDF'
    df.loc[df['PWF'] == 1, 'Failure type'] = 'PWF'
    df.loc[df['OSF'] == 1, 'Failure type'] = 'OSF'
    df.loc[df['RNF'] == 1, 'Failure type'] = 'RNF'
    failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for index, row in df.iterrows():
        if row['Failure type'] == 'No Failure' and row['Machine failure'] == 1:
            for col in failure_columns:
                if row[col] == 1:
                    df.loc[index, 'Failure type'] = col
                    break
    df = df.drop(columns=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    return df

def label_enc(df):
    machine_type_dict = {'L': 0, 'M': 1, 'H': 2}
    machine_failure_dict = {'No Failure': 0, 'PWF': 1, 'OSF': 2, 'HDF': 3, 'TWF': 4, 'RNF': 5}
    df['Machine type'].replace(to_replace=machine_type_dict, inplace=True)
    df['Failure type'].replace(to_replace=machine_failure_dict, inplace=True)
    return df

def scaling_data(df, float_cols):
    sc = StandardScaler()
    df[float_cols] = sc.fit_transform(df[float_cols])
    return df, sc

def target_anomaly(df):
    anomaly1 = ((df["Machine failure"] == 0) & (df["Failure type"] == 5))
    df.drop(index=df.loc[anomaly1].index, inplace=True)
    anomaly2 = ((df["Machine failure"] == 1) & (df["Failure type"] == 0))
    df.drop(index=df.loc[anomaly2].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def drop_infrequent(df, threshold):
    num_rows = len(df)
    value_counts = df['Failure type'].value_counts(normalize=True) * 100
    infrequent_values = value_counts[value_counts <= threshold].index.tolist()
    df = df[~df['Failure type'].isin(infrequent_values)].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def augment_data(df):
    working = df['Failure type'].value_counts()[0]
    desired_length = round(working / 0.8)
    spc = round((desired_length - working) / 4)
    balance_cause = {0: working, 1: spc, 2: spc, 3: spc, 4: spc}
    aug = SMOTENC(categorical_features=[0, 7], sampling_strategy=balance_cause, random_state=0)
    df, _= aug.fit_resample(df, df['Failure type'])
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest='input_data', type=str, required=True,
                        help="Path to the input data (can be a local path or Azure ML URI)")
    parser.add_argument("--output_data", dest='output_data', type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument("--scaler_output_data", dest='scaler_output_data', type=str, required=True,
                        help="Full path for the StandardScaler object file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
