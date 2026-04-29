# src/models/train.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class ModelTrainer:

    def __init__(self, experiment_name: str = "credit-risk-scoring"):
        mlflow.set_experiment(experiment_name)
        self.model      = None
        self.run_id     = None
        self.best_auc   = 0

    def get_params(self) -> dict:
        """
        LightGBM Parameters — Kya karta hai har ek?
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        n_estimators     : Kitne trees banane hain
        learning_rate    : Har tree kitna seekhe
                           (chota = slow but better)
        num_leaves       : Tree ki complexity
                           (zyada = overfit risk)
        min_child_samples: Leaf mein minimum samples
                           (overfit rokta hai)
        subsample        : Har tree ke liye
                           data ka % use karo
        colsample_bytree : Har tree ke liye
                           features ka % use karo
        reg_alpha/lambda : Regularization
                           (overfit rokta hai)
        scale_pos_weight : class imbalance weight
                           = non_default/default count
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return {
            'n_estimators'     : 1000,
            'learning_rate'    : 0.05,
            'num_leaves'       : 63,
            'max_depth'        : -1,
            'min_child_samples': 50,
            'subsample'        : 0.8,
            'colsample_bytree' : 0.8,
            'reg_alpha'        : 0.1,
            'reg_lambda'       : 0.1,
            'scale_pos_weight' : 2,    # After SMOTE (1:2 ratio)
            'n_jobs'           : -1,   # All CPU cores use karo
            'verbose'          : -1    # Logs suppress karo
        }

    def train(self,
              X_train, y_train,
              X_test,  y_test,
              run_name: str = "lgbm_baseline"):

        params = self.get_params()

        print(f"\n🚀 Starting MLflow Run: {run_name}")
        print("=" * 50)

        with mlflow.start_run(run_name=run_name) as run:
            self.run_id = run.info.run_id

            # ── Step 1: Log Parameters ──
            mlflow.log_params(params)

            # ── Step 2: Train Model ──
            self.model = lgb.LGBMClassifier(**params, random_state=42)

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50,
                                       verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )

            # ── Step 3: Predict ──
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred       = self.model.predict(X_test)

            # ── Step 4: Calculate Metrics ──
            roc_auc  = roc_auc_score(y_test, y_pred_proba)
            pr_auc   = average_precision_score(y_test, y_pred_proba)
            best_itr = self.model.best_iteration_

            print(f"\n📊 RESULTS:")
            print(f"ROC-AUC  : {roc_auc:.4f}")
            print(f"PR-AUC   : {pr_auc:.4f}")
            print(f"Best Iter: {best_itr}")

            # ── Step 5: Log Metrics to MLflow ──
            mlflow.log_metric("roc_auc",        roc_auc)
            mlflow.log_metric("pr_auc",          pr_auc)
            mlflow.log_metric("best_iteration",  best_itr)

            # ── Step 6: Confusion Matrix Plot ──
            self._plot_confusion_matrix(y_test, y_pred, run)

            # ── Step 7: Feature Importance Plot ──
            self._plot_feature_importance(X_train.columns, run)

            # ── Step 8: Log Model to MLflow ──
            mlflow.lightgbm.log_model(
                self.model,
                artifact_path="lgbm_model"
            )

            # ── Step 9: Save locally too ──
            os.makedirs('artifacts', exist_ok=True)
            joblib.dump(self.model, 'artifacts/lgbm_model.pkl')
            mlflow.log_artifact('artifacts/lgbm_model.pkl')

            self.best_auc = roc_auc
            print(f"\n✅ Run complete! Run ID: {self.run_id[:8]}...")

        return self.model, y_pred_proba

    def _plot_confusion_matrix(self, y_test, y_pred, run):
        """MLflow mein confusion matrix save karo"""
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt='d',
            cmap='Blues', ax=ax,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default']
        )
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=13)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

        # Calculate rates
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix:")
        print(f"  True Negative  (Correctly said No Default): {tn:,}")
        print(f"  True Positive  (Correctly said Default)   : {tp:,}")
        print(f"  False Positive (Wrong Default alert)      : {fp:,}")
        print(f"  False Negative (Missed Default!)          : {fn:,}")

        path = 'data/processed/confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        mlflow.log_artifact(path)
        plt.show()

    def _plot_feature_importance(self, feature_names, run):
        """Top 15 important features plot"""
        importance = pd.DataFrame({
            'feature'   : feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(
            importance['feature'],
            importance['importance'],
            color='#3498db',
            edgecolor='white'
        )
        ax.set_title('Top 15 Feature Importances (LightGBM)',
                     fontweight='bold', fontsize=13)
        ax.set_xlabel('Importance Score')
        ax.invert_yaxis()

        path = 'data/processed/feature_importance.png'
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        mlflow.log_artifact(path)
        plt.show()


# ─────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────
if __name__ == "__main__":

    # Load data
    X_train = pd.read_csv('data/splits/X_train_balanced.csv')
    y_train = pd.read_csv('data/splits/y_train_balanced.csv').squeeze()
    X_test  = pd.read_csv('data/splits/X_test.csv')
    y_test  = pd.read_csv('data/splits/y_test.csv').squeeze()

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    trainer = ModelTrainer()
    model, y_pred_proba = trainer.train(
        X_train, y_train,
        X_test,  y_test,
        run_name="lgbm_baseline_v1"
    )

    # Save predictions for Day 4
    pd.Series(y_pred_proba).to_csv(
        'data/splits/y_pred_proba.csv', index=False
    )
    print("\n✅ Day 3 Complete — Model trained and logged!")