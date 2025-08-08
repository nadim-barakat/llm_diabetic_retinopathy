
# DR MLLM Results Summary

This file summarizes results for **Experiment 1 (Zero-shot)**, **Experiment 2 (LLM + CNN integration)**, and **Experiment 3 (Model-to-Model prompting)** across the **IDRiD** and **Messidor** datasets.

---

## Experiment 1 — Zero-shot

### IDRiD
| Dataset | Model  | Variant   | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---:|---:|---:|---:|
| IDRiD | GPT-4O   | Zero-Shot | 0.69 | 0.50 | 1.00 | 0.78 |
| IDRiD | MedGemma | Zero-Shot | 0.73 | 0.92 | 0.41 | 0.87 |

### Messidor
| Dataset | Model  | Variant   | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---:|---:|---:|---:|
| Messidor | GPT-4O   | Zero-Shot | 0.64 | 0.40 | 1.00 | 0.75 |
| Messidor | MedGemma | Zero-Shot | 0.87 | 0.95 | 0.76 | 0.95 |

---

## Experiment 2 — Adding CNN

### IDRiD
| Dataset | Model  | Variant | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---:|---:|---:|---:|
| IDRiD | GPT-4O   | + CNN (70%) | 0.75 | 0.59 | 1.00 | 0.96 |
| IDRiD | GPT-4O   | + CNN (80%) | 0.76 | 0.61 | 1.00 | 0.98 |
| IDRiD | GPT-4O   | + CNN (90%) | 0.79 | 0.66 | 1.00 | 0.98 |
| IDRiD | GPT-4O   | + CNN (70%) — False answer | 0.41 | 0.06 | 0.97 | 0.61 |
| IDRiD | GPT-4O   | + CNN (80%) — False answer | 0.39 | 0.03 | 0.97 | 0.56 |
| IDRiD | GPT-4O   | + CNN (90%) — False answer | 0.39 | 0.03 | 0.97 | 0.49 |
| IDRiD | MedGemma | + CNN (70%) | 0.94 | 1.00 | 0.85 | 0.95 |
| IDRiD | MedGemma | + CNN (80%) | 0.95 | 1.00 | 0.87 | 0.96 |
| IDRiD | MedGemma | + CNN (90%) | 0.93 | 0.98 | 0.85 | 0.96 |
| IDRiD | MedGemma | + CNN (70%) — False answer | 0.54 | 0.73 | 0.23 | 0.77 |
| IDRiD | MedGemma | + CNN (80%) — False answer | 0.54 | 0.77 | 0.18 | 0.78 |
| IDRiD | MedGemma | + CNN (90%) — False answer | 0.55 | 0.77 | 0.21 | 0.76 |

### Messidor
| Dataset | Model  | Variant | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---:|---:|---:|---:|
| Messidor | GPT-4O   | + CNN (70%) | 0.77 | 0.61 | 1.00 | 0.95 |
| Messidor | GPT-4O   | + CNN (80%) | 0.79 | 0.66 | 1.00 | 0.96 |
| Messidor | GPT-4O   | + CNN (90%) | 0.79 | 0.66 | 1.00 | 0.96 |
| Messidor | GPT-4O   | + CNN (70%) — False answer | 0.46 | 0.12 | 0.96 | 0.58 |
| Messidor | GPT-4O   | + CNN (80%) — False answer | 0.44 | 0.10 | 0.96 | 0.57 |
| Messidor | GPT-4O   | + CNN (90%) — False answer | 0.41 | 0.05 | 0.96 | 0.53 |
| Messidor | MedGemma | + CNN (70%) | 0.98 | 1.00 | 0.94 | 1.00 |
| Messidor | MedGemma | + CNN (80%) | 0.98 | 0.99 | 0.97 | 0.99 |
| Messidor | MedGemma | + CNN (90%) | 0.98 | 0.99 | 0.96 | 0.99 |
| Messidor | MedGemma | + CNN (70%) — False answer | 0.67 | 0.82 | 0.44 | 0.88 |
| Messidor | MedGemma | + CNN (80%) — False answer | 0.64 | 0.79 | 0.41 | 0.87 |
| Messidor | MedGemma | + CNN (90%) — False answer | 0.64 | 0.78 | 0.43 | 0.87 |

---

## Experiment 3 — Model-to-Model

### IDRiD
| Dataset | Model | Variant | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---:|---:|---:|---:|
| IDRiD | GPT-4O | Given MedGemma binary predictions only | 0.73 | 0.58 | 0.97 | 0.85 |
| IDRiD | GPT-4O | Given MedGemma descriptions only | 0.78 | 0.86 | 0.64 | 0.90 |
| IDRiD | GPT-4O | Given MedGemma descriptions + binary referral | 0.76 | 0.92 | 0.49 | 0.90 |
| IDRiD | MedGemma→desc, GPT-4O→grade | Description-based (Zero-shot prompt), No image | 0.80 | 0.86 | 0.69 | 0.91 |
| IDRiD | MedGemma→desc, GPT-4O→grade | Description-based (90% CNN prompt), No image | 0.83 | 0.89 | 0.72 | 0.90 |

### Messidor
| Dataset | Model | Variant | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---:|---:|---:|---:|
| Messidor | GPT-4O | Given MedGemma binary predictions only | 0.77 | 0.62 | 1.00 | 0.87 |
| Messidor | GPT-4O | Given MedGemma descriptions only | 0.90 | 0.95 | 0.81 | 0.96 |
| Messidor | GPT-4O | Given MedGemma descriptions + binary referral | 0.89 | 0.95 | 0.79 | 0.96 |
| Messidor | MedGemma→desc, GPT-4O→grade | Description-based (Zero-shot), No image | 0.90 | 0.92 | 0.87 | 0.96 |
| Messidor | MedGemma→desc, GPT-4O→grade | Description-based (90% CNN), No image | 0.92 | 0.93 | 0.90 | 0.98 |

---

### Notes
- “False answer” rows indicate the CNN-provided binary input was intentionally incorrect while keeping the stated sensitivity/specificity..

