
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# category_encoders is needed for Target Encoding text

try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will not be available.")


def main():

    print("Loading Dataset:")
    file_path = "insurance.csv"

    if not os.path.exists(file_path):
        print(f"Error: Cannot find '{file_path}'")
        return

    df = pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully, Rows: {df.shape[0]}, Features: {df.shape[1]}\n")


    # Handling Missing Values
    print("\nHandling Missing Values:\n")
    print("Artificially creating missing data for the issues")

    df.loc[0:24, 'bmi'] = np.nan

    imputer = SimpleImputer(strategy='median')

    df['bmi'] = imputer.fit_transform(df[['bmi']]).ravel()

    print(f"Imputation complete. 'bmi' now has {df['bmi'].isnull().sum()} null values.\n")

    print("Evaluating the skewness of the charges distribution...")

    df['LogCharges'] = np.log1p(df['charges'])

    print(f"Log Transformation applied. New skewness: {df['LogCharges'].skew():.2f} (closer to 0 is perfectly balanced).\n")



    if TargetEncoder is not None:
        print("Applying Target Encoder")

        encoder = TargetEncoder()

        df["region_encoded"] = encoder.fit_transform(df["region"], df["charges"])

    else:
        print("Category Encoder not installed")

    print("\nFeature Selection:\n")

    # Feature Selection

    features_to_test = ['age', 'bmi', 'children']

    X_features = df[features_to_test].fillna(0)
    y_target = df['charges']

    selector = SelectKBest(score_func=mutual_info_regression, k=2)

    selector.fit(X_features, y_target)

    winning_features = selector.get_support()
    best_features = X_features.columns[winning_features].tolist()

    print(best_features)

    print("\nSpliting Data:\n")

    # Spliting Data

    X = df[best_features]
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training Data Size: {X_train.shape}")
    print(f"Testing Data Size: {X_test.shape}\n")

    print("\nTraining Model:\n")

    # Training Model

    model=LinearRegression()
    model.fit(X_train,y_train)

    prediction = model.predict(X_test)
    print(prediction)


    print("\nComparing model prediction to the actual real answer:\n")


    # Comparing model prediction to the actual real answer

    actual_wins = y_test.head(3).values
    predicted_wins=prediction[:3]

    for i in range(3):
        predicted = round(predicted_wins[i])
        actual = actual_wins[i]
        difference=abs(actual-predicted)
        print(f"Model Gussed:{predicted}")
        print(f"Real Answer:{actual}")
        print(f"Differences:{difference}")


if __name__ == "__main__":
    main()