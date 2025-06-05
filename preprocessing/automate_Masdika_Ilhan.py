# preprocessing/automate_Masdika_Ilhan.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import os
from io import StringIO

def clean_csv_file(input_path):
    """Normalize line endings and fix malformed lines"""
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        content = f.read()
    
    # Normalize line endings and filter valid lines
    lines = []
    for line in content.splitlines():
        if line.count(',') == 8:  # Sesuai jumlah kolom dataset diabetes
            lines.append(line)
    
    return StringIO('\n'.join(lines))

def preprocess_data(input_path, output_path):
    # Buat direktori output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Baca dan bersihkan file
    clean_data = clean_csv_file(input_path)
    
    # Load ke DataFrame
    df = pd.read_csv(clean_data)

    # # Load data
    # df = pd.read_csv(input_path)
    
    # 1. Handle zero values (sesuai notebook)
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_features] = df[zero_features].replace(0, np.nan)
    
    # 2. Impute missing values
    imputer = SimpleImputer(strategy='median')
    df[zero_features] = imputer.fit_transform(df[zero_features])
    
    # 3. Split features & target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # 4. Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save processed data (gabungkan train dan test)
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['Outcome'] = y_train.values
    train_df['Data_Type'] = 'train'
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['Outcome'] = y_test.values
    test_df['Data_Type'] = 'test'
    
    combined_df = pd.concat([train_df, test_df])
    combined_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/diabetes.csv", help="Path to raw data")
    parser.add_argument("--output", type=str, default="data/processed/diabetes_processed.csv", help="Path to save processed data")
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)