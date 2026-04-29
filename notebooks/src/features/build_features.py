# ============================================
# src/features/build_features.py
# ============================================

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Domain knowledge → New Features

    WHY domain knowledge zaroori hai?
    Raw features → Model
    Engineer features → Smarter Model

    Jaise: Ek banker kaise sochta hai?
    → Debt kitna hai income ke relative?
    → Kitni baar late payments ki?
    → Credit cards kitne use kar raha hai?
    """

    def add_debt_to_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DTI = Debt Ratio × Income
        Actual monthly debt payment estimate
        Banks ka #1 metric hai yeh
        """
        df['MonthlyDebtPayment'] = df['DebtRatio'] * df['MonthlyIncome']

        df['DebtToIncomeCategory'] = pd.cut(
            df['DebtRatio'],
            bins=[0, 0.2, 0.4, 0.6, float('inf')],
            labels=['Low', 'Medium', 'High', 'VeryHigh']
        ).astype(str)

        return df

    def add_payment_history_features(self,
                                     df: pd.DataFrame) -> pd.DataFrame:
        """
        3 alag late payment columns hain
        Inhe ek powerful feature mein combine karo
        """
        late_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]

        # Total late payments (simple sum)
        df['TotalLatePayments'] = df[late_cols].sum(axis=1)

        # Weighted late payments (90 din late = 3x bura)
        df['WeightedLatePayments'] = (
            df['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
            df['NumberOfTime60-89DaysPastDueNotWorse'] * 2 +
            df['NumberOfTimes90DaysLate']               * 3
        )

        # Binary: Kya kabhi bhi late hua?
        df['EverLate'] = (df['TotalLatePayments'] > 0).astype(int)

        return df

    def add_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Age ek continuous feature hai
        Age groups mein convert karo — nonlinear patterns capture hote hain
        """
        df['AgeBucket'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['VeryYoung','Young','Middle','MidSenior','Senior','Elder']
        ).astype(str)

        # Young + high debt = highest risk combination
        df['YoungHighDebt'] = (
            (df['age'] < 30) &
            (df['DebtRatio'] > 0.4)
        ).astype(int)

        return df

    def add_credit_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Credit utilization > 30% = bad credit signal
        > 70% = very risky
        """
        util = df['RevolvingUtilizationOfUnsecuredLines']

        df['UtilizationCategory'] = pd.cut(
            util,
            bins=[-0.01, 0.3, 0.7, 1.0, float('inf')],
            labels=['Good', 'Fair', 'Poor', 'Critical']
        ).astype(str)

        # Binary high utilization flag
        df['HighUtilization'] = (util > 0.7).astype(int)

        return df

    def add_income_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Income per dependent = actual financial burden
        """
        df['IncomePerDependent'] = (
            df['MonthlyIncome'] /
            (df['NumberOfDependents'] + 1)  # +1 for self
        )

        df['LowIncomeFlag'] = (
            df['MonthlyIncome'] < 3000
        ).astype(int)

        return df

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Master function — ek hi call mein sab features"""
        print("🔨 Building features...")
        df = self.add_debt_to_income(df)
        df = self.add_payment_history_features(df)
        df = self.add_age_features(df)
        df = self.add_credit_utilization(df)
        df = self.add_income_features(df)
        print(f"✅ Features added | New shape: {df.shape}")
        return df


# ---- USAGE ----
if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_data.csv')

    fe = FeatureEngineer()
    df_featured = fe.build_all(df)

    # Save
    df_featured.to_csv(
        'data/processed/featured_data.csv',
        index=False
    )
    print("\n📊 New features created:")
    new_cols = [
        'MonthlyDebtPayment', 'DebtToIncomeCategory',
        'TotalLatePayments', 'WeightedLatePayments', 'EverLate',
        'AgeBucket', 'YoungHighDebt',
        'UtilizationCategory', 'HighUtilization',
        'IncomePerDependent', 'LowIncomeFlag'
    ]
    print(df_featured[new_cols].head(3))