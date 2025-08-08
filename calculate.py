import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os

import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error
)
from scipy.stats import pearsonr, spearmanr
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import Counter

MEDGEMMA = True
MESSIDOR = True

with open("gpt4o_messidor_90______medgemma_ref_descrip_no_image.json", "r") as f:
    predictions = json.load(f)

if MESSIDOR:
    truth_df = pd.read_csv("/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/messidor_subset/messidor_subset_175_balanced_cleaned.csv")
else:
    truth_df = pd.read_csv("/Users/nadim/Documents/Tufts/medical_school /research/dana_farber/ophtho_mllm/archive/b_disease_grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

truth_lookup = {
    row["Image name"]: int(row["Retinopathy grade"])
    for _, row in truth_df.iterrows()
}

y_true = []
y_pred = []

for pred in predictions:
    img_name = os.path.splitext(pred["image_name"])[0]
    llm_data = pred["llm_response"]
    
    
    true_grade = truth_lookup[img_name]
    true_referral = 1 if true_grade >= 2 else 0

    # Get model's predicted referral
    predicted_referral = llm_data.get("referral", None)
    

    y_true.append(true_referral)
    y_pred.append(predicted_referral)


# Generate report
print("Binary Referral Classification Report (based on `referral` field):")
print(classification_report(y_true, y_pred, target_names=["No Referral", "Referral"]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)


tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}\n\n")


matched_results = []
for item in predictions:
    img_name = os.path.splitext(item["image_name"])[0]
    # predicted_grade = item["llm_response"]["grade"]
    predicted_grade = item["llm_response"]["score"]

    # referral_prediction = item["llm_response"]["referral"]
    # score_prediction = item["llm_response"]["score"]

    if img_name in truth_lookup:
        true_grade = truth_lookup[img_name] * 25  # Scale 0–4 to 0–100
        matched_results.append((img_name, true_grade, predicted_grade))
    else:
        print(f"Warning: {img_name} not found in ground truth.")

# Separate into arrays
ground_truth = np.array([x[1] for x in matched_results])      # 0–4
llm_scores = np.array([x[2] for x in matched_results])        # 0–100, float


true_referral = np.array([1 if x >= 50 else 0 for x in ground_truth])
# llm_referral = np.array([1 if x[2] >= 3 else 0 for x in matched_results])

# print(llm_referral)


fpr, tpr, thresholds = roc_curve(true_referral, llm_scores)
roc_auc = auc(fpr, tpr)

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"Best threshold by Youden's J: {best_threshold:.2f}")

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label='Best threshold')

# Annotate the threshold value
plt.annotate(f'TPR: {tpr[best_idx]:.2f}\nFPR: {fpr[best_idx]:.2f}\nThreshold: {best_threshold:.0f}',
             xy=(fpr[best_idx], tpr[best_idx]),
             xytext=(fpr[best_idx] + 0.05, tpr[best_idx] - 0.1),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10,
             color='red')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

dataset_name = f"{'MedGemma' if MEDGEMMA else 'GPT4o'} On {'MESSIDOR' if MESSIDOR else 'IDRiD'}"

plt.title("GPT4o Referral on Messidor \n with Medgemma's description after 90% CNN prompt \n and no image")
plt.legend(loc='lower right')
plt.grid()
plt.show()

print(f"Best threshold: {best_threshold:.2f}")
print(f"TPR at best threshold: {tpr[best_idx]:.2f}")
print(f"FPR at best threshold: {fpr[best_idx]:.2f}")

# --- Regression Metrics ---
print("Pearson:", pearsonr(ground_truth, llm_scores))
print("Spearman:", spearmanr(ground_truth, llm_scores))
print("MAE:", mean_absolute_error(ground_truth, llm_scores))
rmse = np.sqrt(mean_squared_error(ground_truth, llm_scores))
print("RMSE:", rmse)


score_counts = Counter(llm_scores)
for score, count in sorted(score_counts.items()):
    print(f"Score: {score:.2f}, Count: {count}")