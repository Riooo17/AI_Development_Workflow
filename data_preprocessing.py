import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_student_data(df):
    """
    Preprocess student dropout dataset
    Steps:
        1. Handle missing values
        2. Normalize numeric features
        3. Encode categorical features
    """
    # 1. Impute missing numeric values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # 2. Normalize numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 3. Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/student_data.csv")
    df_processed = preprocess_student_data(df)
    df_processed.to_csv("../data/processed/student_data_processed.csv", index=False)
