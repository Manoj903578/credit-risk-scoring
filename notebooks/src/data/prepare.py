# src/data/prepare.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class DataPreparer:
    """
    WHY alag class?
    - Encoding + Scaling sirf train pe fit hota hai
    - Test pe sirf transform hota hai
    - Production mein same object reuse hoga
    """

    def __init__(self):
        self.label_encoders = {}    # Har categorical column ka alag encoder
        self.scaler         = StandardScaler()
        self.feature_cols   = None
        self.target_col     = 'SeriousDlqin2yrs'

    def encode_categoricals(self,
                             df: pd.DataFrame,
                             fit: bool = True) -> pd.DataFrame:
        """
        Categorical columns → Numbers
        WHY? LightGBM numbers chahta hai, strings nahi

        Example:
        'AgeBucket': Young → 0, Middle → 1, Senior → 2
        """
        df = df.copy()

        # Categorical columns dhundo
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"📝 Categorical columns: {cat_cols}")

        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le     # Save for later
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))

        print(f"✅ Encoded {len(cat_cols)} categorical columns")
        return df

    def split_data(self, df: pd.DataFrame):
        """
        80% Training, 20% Testing
        WHY Stratify?
        - Target 93:7 imbalanced hai
        - Stratify ensure karta hai dono sets mein
          same ratio ho
        """
        # Target aur features alag karo
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        self.feature_cols = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y          # ← IMPORTANT for imbalanced data
        )

        print(f"\n📊 SPLIT SUMMARY:")
        print(f"Training set   : {X_train.shape[0]:,} rows")
        print(f"Test set       : {X_test.shape[0]:,} rows")
        print(f"\nTarget in Train:")
        print(f"  Non-Default: {sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.1f}%)")
        print(f"  Default    : {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.1f}%)")

        return X_train, X_test, y_train, y_test

    def scale_features(self,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        fit: bool = True):
        """
        WHY Scaling?
        LightGBM ke liye strictly zaruri nahi
        Lekin kuch features (income vs age) ka
        scale bahut alag hota hai
        Future mein logistic regression bhi try
        kar sako — tab scaling zaroori hogi

        StandardScaler: mean=0, std=1 karta hai
        """
        if fit:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_scaled = X_train

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        print(f"✅ Features scaled")
        return X_train_scaled, X_test_scaled

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"💾 DataPreparer saved → {path}")

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


# ─────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────
if __name__ == "__main__":

    # Load featured data
    df = pd.read_csv('data/processed/featured_data.csv')
    print(f"Loaded shape: {df.shape}")

    preparer = DataPreparer()

    # Step 1: Encode
    df_encoded = preparer.encode_categoricals(df, fit=True)

    # Step 2: Split
    X_train, X_test, y_train, y_test = preparer.split_data(df_encoded)

    # Step 3: Scale
    X_train_sc, X_test_sc = preparer.scale_features(X_train, X_test, fit=True)

    # Save splits
    os.makedirs('data/splits', exist_ok=True)
    X_train_sc.to_csv('data/splits/X_train.csv', index=False)
    X_test_sc.to_csv('data/splits/X_test.csv',   index=False)
    y_train.to_csv('data/splits/y_train.csv',     index=False)
    y_test.to_csv('data/splits/y_test.csv',       index=False)

    # Save preparer
    preparer.save('artifacts/data_preparer.pkl')

    print("\n✅ Day 1 Complete!")
    print(f"X_train: {X_train_sc.shape} | X_test: {X_test_sc.shape}")