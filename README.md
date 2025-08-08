# DR MLLM Experiments

This repository tests **GPT-4o** and **MedGemma** for detecting **referable diabetic retinopathy (DR)**.  
Each model was prompted to generate:  
1. A **description** of the image  
2. A **referral decision** (yes/no)
3. A **continuous DR severity score** (0–100)

Multiple experimental setups were used, controlled via **command-line arguments**.

---

## Datasets

- **IDRiD** — 103 images  
- **Messidor-2** — balanced subset of 175 images (35 per category)

---

## Model Setup

- **MedGemma** — run locally via `llama.cpp`  
  - Download MedGemma weights and update `medgemma.py` to point to your `.gguf` files.
- **GPT-4o** — run through the OpenAI API (requires API key).

---

## Experiments

### **Experiment 1 — Zero-shot**
Run models without additional context.

**Examples:**
```bash
# IDRiD — GPT-4o
python local_eye_grade.py --end_index 103

# IDRiD — MedGemma
python local_eye_grade.py --medgemma --end_index 103

# Messidor — GPT-4o
python local_eye_grade.py --messidor

# Messidor — MedGemma
python local_eye_grade.py --messidor --medgemma
```

---

### **Experiment 2 — Simulated CNN Integration**
Simulates a CNN providing binary outputs (true or false) at different accuracies (70%, 80%, 90%) to see how model performance changes.  
Requires `--external_help`.

**Examples:**
```bash
# Messidor — MedGemma with false CNN answer
python local_eye_grade.py --external_help --accuracy 70 --medgemma --messidor --opposite

# Messidor — MedGemma with correct CNN answer
python local_eye_grade.py --external_help --accuracy 70 --medgemma --messidor

# IDRiD — GPT-4o with false CNN answer
python local_eye_grade.py --external_help --accuracy 80 --opposite --end_index 103
```

---

### **Experiment 3 — Model-to-Model Prompting**
Uses MedGemma outputs as GPT-4o inputs under different scenarios:  
- Image + binary result  
- Image + binary + description  
- Image + description only  
- Description only (no image)

Requires `--medgemma_ref`.

**Examples:**
```bash
# Messidor — GPT-4o with image + MedGemma binary
python local_eye_grade.py --medgemma_ref --binary_only

# IDRiD — GPT-4o with image + MedGemma descriptions
python local_eye_grade.py --medgemma_ref --descrip_only --end_index 103

# Messidor — GPT-4o with MedGemma descriptions only (no image)
python local_eye_grade.py --medgemma_ref --descrip_no_image --messidor
```

---

## Processing Results

After running an experiment, you’ll get a `.jsonl` file.  
To compute **accuracy, sensitivity, specificity, and AUC**:

1. Convert to `.json`:
```bash
python jsonl-json.py your_file.jsonl
```
2. Update `calculate.py` to point to the new `.json` file.

---

## Error Handling

If an image fails to process, the script will automatically retry **once**.  
If it still fails, the error is written to `error_log.txt`.  
You can take the failed filenames from this log, place them in a file called `batch.txt`, and re-run only those images using the `--batch` flag:

```bash
python local_eye_grade.py --[all your other flags] --batch
```

---

## Output

- **Actual Results:** see [`results.md`](./results.md) for full metrics from all experiments.  
- **Raw outputs:** individual JSONs for each run are in `/updated_output`.
