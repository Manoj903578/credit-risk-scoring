# src/models/explain.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib


class ModelExplainer:

    def __init__(self, model, X_train: pd.DataFrame):
        print("⏳ Building SHAP explainer...")

        self.model = model
        self.X_train = X_train

        # Use NEW SHAP API
        self.explainer = shap.Explainer(model)

        # Sample data (performance optimization)
        sample = X_train.sample(min(2000, len(X_train)), random_state=42)

        # Compute SHAP values
        self.shap_values = self.explainer(sample)
        self.X_sample = sample

        print("✅ SHAP Explainer ready!")

    # ─────────────────────────────────────
    # INTERNAL HELPER (VERY IMPORTANT)
    # ─────────────────────────────────────
    def _get_shap_matrix(self, shap_values):
        """
        Converts SHAP output → 2D matrix (safe for plots)
        Handles:
        - Regression
        - Binary classification
        - Multi-class
        """
        values = shap_values.values

        # Case: multi-class (3D → [samples, features, classes])
        if len(values.shape) == 3:
            values = values[:, :, 1]  # take class 1

        return values

    # ─────────────────────────────────────
    # GLOBAL IMPORTANCE (BAR)
    # ─────────────────────────────────────
    def plot_global_importance(self):
        values = self._get_shap_matrix(self.shap_values)

        plt.figure(figsize=(10, 7))

        shap.summary_plot(
            values,
            self.X_sample,
            plot_type="bar",
            show=False,
            max_display=15
        )

        plt.title('Global Feature Importance (SHAP)',
                  fontweight='bold', fontsize=13)

        plt.tight_layout()
        plt.savefig('data/processed/shap_global.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        print("✅ Global SHAP plot saved!")

    # ─────────────────────────────────────
    # BEESWARM PLOT
    # ─────────────────────────────────────
    def plot_beeswarm(self):
        values = self._get_shap_matrix(self.shap_values)

        plt.figure(figsize=(10, 8))

        shap.summary_plot(
            values,
            self.X_sample,
            show=False,
            max_display=15
        )

        plt.title('SHAP Beeswarm — Feature Impact Distribution',
                  fontweight='bold', fontsize=13)

        plt.tight_layout()
        plt.savefig('data/processed/shap_beeswarm.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

    # ─────────────────────────────────────
    # SINGLE CUSTOMER EXPLANATION
    # ─────────────────────────────────────
    def explain_single_customer(self,
                                customer_data: pd.DataFrame,
                                customer_id: str = "CUST_001"):

        print(f"\n🔍 Explaining prediction for: {customer_id}")

        shap_vals = self.explainer(customer_data)
        values = self._get_shap_matrix(shap_vals)

        # Base value (safe)
        base_value = shap_vals.base_values[0]

        print(f"Base (avg prediction): {base_value:.4f}")

        explanation = shap.Explanation(
            values=values[0],
            base_values=base_value,
            data=customer_data.iloc[0].values,
            feature_names=customer_data.columns.tolist()
        )

        plt.figure(figsize=(12, 6))

        shap.plots.waterfall(
            explanation,
            show=False,
            max_display=12
        )

        plt.title(f'SHAP Waterfall — {customer_id}',
                  fontweight='bold', fontsize=13)

        plt.tight_layout()
        plt.savefig(f'data/processed/shap_waterfall_{customer_id}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        # Human-readable explanation
        feature_impacts = sorted(
            zip(customer_data.columns, values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        print(f"\n📋 TOP FACTORS for {customer_id}:")
        print("-" * 50)

        for feat, impact in feature_impacts[:5]:
            direction = "⬆️ INCREASES" if impact > 0 else "⬇️ DECREASES"
            print(f"{direction} risk | {feat}: {impact:+.4f}")

        return shap_vals

    # ─────────────────────────────────────
    # SAVE EXPLAINER
    # ─────────────────────────────────────
    def save(self, path: str):
        joblib.dump(self, path)
        print(f"💾 Explainer saved → {path}")


# ─────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────
if __name__ == "__main__":

    # Load model + data
    model = joblib.load('artifacts/lgbm_model.pkl')
    X_train = pd.read_csv('data/splits/X_train_balanced.csv')
    X_test = pd.read_csv('data/splits/X_test.csv')

    # Build explainer
    explainer = ModelExplainer(model, X_train)

    # Global plots
    explainer.plot_global_importance()
    explainer.plot_beeswarm()

    # Explain customers
    high_risk = X_test.iloc[[0]]
    explainer.explain_single_customer(high_risk, "HIGH_RISK_CUST")

    low_risk = X_test.iloc[[5]]
    explainer.explain_single_customer(low_risk, "LOW_RISK_CUST")

    # Save
    explainer.save('artifacts/shap_explainer.pkl')

    print("\n✅ SHAP Explainability Completed Successfully!")