# src/data/balance.py

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series,
                sampling_strategy: float = 0.5):
    """
    sampling_strategy=0.5 means:
    Default : Non-Default = 1:2 ratio
    (50% of majority class)

    WHY not 1:1?
    Full balance sometimes causes overfitting
    1:2 ratio works well in practice
    """

    print("⏳ Applying SMOTE...")
    print(f"Before SMOTE: {Counter(y_train)}")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=5      # Nearest neighbors for synthetic sample
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE : {Counter(y_resampled)}")
    print(f"New training size: {X_resampled.shape[0]:,}")

    # Visual comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Before
    before_counts = Counter(y_train)
    axes[0].bar(['Non-Default', 'Default'],
                [before_counts[0], before_counts[1]],
                color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('BEFORE SMOTE', fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate([before_counts[0], before_counts[1]]):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

    # After
    after_counts = Counter(y_resampled)
    axes[1].bar(['Non-Default', 'Default'],
                [after_counts[0], after_counts[1]],
                color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('AFTER SMOTE', fontweight='bold')
    axes[1].set_ylabel('Count')
    for i, v in enumerate([after_counts[0], after_counts[1]]):
        axes[1].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

    plt.suptitle('SMOTE — Class Balancing Effect',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/processed/smote_comparison.png', dpi=150)
    plt.show()

    return pd.DataFrame(X_resampled,
                        columns=X_train.columns), \
           pd.Series(y_resampled, name='SeriousDlqin2yrs')


# ─────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────
if __name__ == "__main__":

    X_train = pd.read_csv('data/splits/X_train.csv')
    y_train = pd.read_csv('data/splits/y_train.csv').squeeze()

    X_bal, y_bal = apply_smote(X_train, y_train, sampling_strategy=0.5)

    # Save balanced data
    X_bal.to_csv('data/splits/X_train_balanced.csv', index=False)
    y_bal.to_csv('data/splits/y_train_balanced.csv', index=False)

    print("\n✅ Day 2 Complete — Balanced data saved!")