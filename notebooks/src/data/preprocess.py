# ============================================
# src/data/preprocess.py
# ============================================

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class DataPreprocessor:
    """
    WHY class banaya?
    - Train pe fit, test pe sirf transform
    - Imputer values save karke reuse kar sako
    - Production mein same preprocessing lagani padti hai
    """

    def __init__(self):
        self.income_imputer  = SimpleImputer(strategy='median')
        self.is_fitted       = False

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        print(f"🗑️ Duplicates removed: {before - after}")
        return df

    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        # Age float hai — int hona chahiye
        df['age'] = df['age'].astype(int)

        # Dependents float — int hona chahiye
        df['NumberOfDependents'] = \
            df['NumberOfDependents'].fillna(0).astype(int)

        print("✅ Data types fixed")
        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        WHY clip? Outliers model ko ek direction mein
        kheeench lete hain. Realistic bounds lagate hain.
        """
        # Age: 0 ya 120+ unrealistic
        df = df[df['age'].between(18, 100)].copy()

        # RevolvingUtilization: 0-1 hona chahiye logically
        # Lekin kuch values 9999 tak hain — clearly errors
        df['RevolvingUtilizationOfUnsecuredLines'] = \
            df['RevolvingUtilizationOfUnsecuredLines'].clip(0, 5)

        # DebtRatio: extreme outliers cap karo
        df['DebtRatio'] = df['DebtRatio'].clip(0, 50)

        # MonthlyIncome: $500k+ is data entry error mostly
        df['MonthlyIncome'] = df['MonthlyIncome'].clip(0, 100000)

        print("✅ Outliers handled")
        return df

    def impute_missing(self,
                       df: pd.DataFrame,
                       fit: bool = True) -> pd.DataFrame:
        """
        fit=True  → Training data pe use karo
        fit=False → Test/new data pe use karo

        WHY alag?
        Test data ki statistics se impute karna
        DATA LEAKAGE hai — never do this!
        """
        df = df.copy()

        if fit:
            df['MonthlyIncome'] = self.income_imputer.fit_transform(
                df[['MonthlyIncome']]
            ).flatten()
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Fit preprocessor first on training data!")
            df['MonthlyIncome'] = self.income_imputer.transform(
                df[['MonthlyIncome']]
            ).flatten()

        # Dependents: missing = 0 (safe assumption)
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

        print(f"✅ Missing values imputed")
        print(f"   Remaining nulls: {df.isnull().sum().sum()}")
        return df

    def save(self, path: str):
        """Preprocessor save karo — production mein reuse hoga"""
        joblib.dump(self, path)
        print(f"💾 Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


# ---- USAGE ----
if __name__ == "__main__":
    df = pd.read_csv('data/raw/cs-training.csv', index_col=0)

    preprocessor = DataPreprocessor()

    df = preprocessor.remove_duplicates(df)
    df = preprocessor.fix_data_types(df)
    df = preprocessor.handle_outliers(df)
    df = preprocessor.impute_missing(df, fit=True)

    # Save cleaned data
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    preprocessor.save('artifacts/preprocessor.pkl')

    print(f"\n✅ Clean data shape: {df.shape}")
    print(f"✅ Nulls remaining : {df.isnull().sum().sum()}")