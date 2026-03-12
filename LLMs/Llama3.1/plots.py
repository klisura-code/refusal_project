import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
path = "/home/dorde/Desktop/Refusal_project/outputs/llama31-impoliteness/llama31_refusal_metrics_by_impoliteness.csv"
df = pd.read_csv(path)

# Sort by metrics for cleaner visualization
df_sorted_refusal = df.sort_values("refusal_accuracy", ascending=False)
df_sorted_fulfill = df.sort_values("fulfillment_rate", ascending=False)

# ---------- Plot 1: Refusal Accuracy ----------
plt.figure(figsize=(12, 8))
plt.barh(df_sorted_refusal["variant"], df_sorted_refusal["refusal_accuracy"])
plt.xlabel("Refusal Accuracy")
plt.ylabel("Impoliteness")
#plt.xlim(0.6, 0.8)   # <<< added
plt.tight_layout()

refusal_out = "/home/dorde/Desktop/Refusal_project/outputs/llama31-impoliteness/llama31_refusal_accuracy_by_impoliteness.pdf"
plt.savefig(refusal_out, bbox_inches="tight")
plt.close()

# ---------- Plot 2: Fulfillment Rate ----------
plt.figure(figsize=(12, 8))
plt.barh(df_sorted_fulfill["variant"], df_sorted_fulfill["fulfillment_rate"])
plt.xlabel("Fulfillment Rate")
plt.ylabel("Impoliteness")
#plt.xlim(0.10, 0.35)   # <<< added
plt.tight_layout()

fulfill_out = "/home/dorde/Desktop/Refusal_project/outputs/llama31-impoliteness/llama31_fulfillment_rate_by_impoliteness.pdf"
plt.savefig(fulfill_out, bbox_inches="tight")
plt.close()

(refusal_out, fulfill_out)
