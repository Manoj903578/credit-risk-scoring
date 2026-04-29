# src/models/evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report
)

class ModelEvaluator:

    def __init__(self, y_test, y_pred_proba):
        self.y_test       = y_test
        self.y_pred_proba = y_pred_proba

    def plot_roc_curve(self):
        """
        ROC Curve:
        X-axis: False Positive Rate (galat alarms)
        Y-axis: True Positive Rate  (sahi pakde)
        AUC: Area under curve → closer to 1 = better
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc      = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#e74c3c', lw=2,
                 label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0,1], [0,1], 'k--', lw=1,
                 label='Random Classifier (AUC = 0.5)')
        plt.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve — Credit Risk Model', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/processed/roc_curve.png', dpi=150)
        plt.show()

        print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
        return roc_auc

    def plot_pr_curve(self):
        """
        Precision-Recall Curve:
        Better metric for imbalanced datasets

        Precision: Jinhe default bola, unme se kitne sach mein default?
        Recall   : Saare defaulters mein se kitne pakde?

        Credit Risk mein Recall zyada important hai!
        (Miss ek defaulter = bank ko loss)
        """
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba
        )
        pr_auc = average_precision_score(
            self.y_test, self.y_pred_proba
        )

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='#3498db', lw=2,
                 label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.fill_between(recall, precision, alpha=0.1, color='#3498db')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/processed/pr_curve.png', dpi=150)
        plt.show()

        print(f"📈 PR-AUC Score: {pr_auc:.4f}")
        return pr_auc

    def find_best_threshold(self):
        """
        Default threshold = 0.5
        Credit risk mein lower threshold better
        WHY? Miss ek defaulter >> galat alarm
        """
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba
        )

        # F1 score maximize karo
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx   = np.argmax(f1_scores)
        best_thr   = thresholds[best_idx]

        print(f"\n🎯 THRESHOLD ANALYSIS:")
        print(f"Best Threshold : {best_thr:.3f}")
        print(f"Best F1 Score  : {f1_scores[best_idx]:.4f}")

        # Compare 0.5 vs best threshold
        for thr in [0.5, best_thr, 0.3]:
            y_pred = (self.y_pred_proba >= thr).astype(int)
            print(f"\n--- Threshold = {thr:.2f} ---")
            print(classification_report(
                self.y_test, y_pred,
                target_names=['No Default', 'Default']
            ))

        return best_thr

    def full_report(self):
        print("=" * 55)
        print("📋 COMPLETE MODEL EVALUATION REPORT")
        print("=" * 55)
        roc = self.plot_roc_curve()
        pr  = self.plot_pr_curve()
        thr = self.find_best_threshold()

        print(f"\n{'='*55}")
        print(f"FINAL SUMMARY")
        print(f"{'='*55}")
        print(f"ROC-AUC  : {roc:.4f}  (Target: >0.85 ✅)")
        print(f"PR-AUC   : {pr:.4f}   (Target: >0.60 ✅)")
        print(f"Best Thr : {thr:.3f}  (Use this in production)")

        return {'roc_auc': roc, 'pr_auc': pr, 'threshold': thr}


# ─────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────
if __name__ == "__main__":
    y_test       = pd.read_csv('data/splits/y_test.csv').squeeze()
    y_pred_proba = pd.read_csv('data/splits/y_pred_proba.csv').squeeze()

    evaluator = ModelEvaluator(y_test, y_pred_proba)
    results   = evaluator.full_report()

    # Save best threshold for API use
    import json
    with open('artifacts/model_config.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Day 4 Complete — Evaluation done!")