from IPython.core import getipython
from IPython.core import getipython
from re import X
import selectors
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

    print("Loading Dataset")
    file_path = "redwine.csv"

    if not os.path.exists(file_path):
        print(f"Error: Cannot find '{file_path}'")
        return

    df = pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully, Rows: {df.shape[0]}, Features: {df.shape[1]}\n")



    # Handling Missing Values
    print("Artificially creating missing data for the issues")

    df.loc[0:24, 'H'] = np.nan

    imputer = SimpleImputer(strategy='median')

    df['H'] = imputer.fit_transform(df[['H']]).ravel()

    print(f"Imputation complete. 'H' now has {df['H'].isnull().sum()} null values.\n")

    print("Evaluating the skewness of the Runs (R) distribution...")

    df['LogRuns'] = np.log1p(df['R'])

    print(f"Log Transformation applied. New skewness: {df['LogRuns'].skew():.2f} (closer to 0 is perfectly balanced).\n")



    df["Team_ID"] = ["Team_" + str(np.random.randint(1,150)) for _ in range(len(df))]

    if TargetEncoder is not None:
        print("Applying Target Encoder")

        encoder = TargetEncoder()

        df["Team_ID_Encoder"] = encoder.fit_transform(df["Team_ID"], df["H"])

    else:
        print("Category Encoder not installed")

    # Feature Selection

    features_to_test = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']

    X_features = df[features_to_test].fillna(0)
    y_target = df['W']

    selector = SelectKBest(score_func=mutual_info_regression, k=2)

    selector.fit(X_features, y_target)

    winning_features = selector.get_support()
    best_features = X_features.columns[winning_features].tolist()

    print(best_features)

    # Spliting Data

    X = df[best_features]
    y = df['H']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training Data Size: {X_train.shape}")
    print(f"Testing Data Size: {X_test.shape}\n")

    # Training Model

    model=LinearRegression()
    model.fit(X_train,y_train)

    prediction = model.predict(X_test)
    print(prediction)


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