**Purpose**: Quick orientation for AI coding agents working on this repository (course project). Focus on the data-driven SVD-subspace digit classification implemented in `problem1.py`.

**Project Big Picture**
- **Main script**: `problem1.py` — loads datasets, computes per-digit SVDs, classifies test samples by projecting onto digit subspaces, and plots accuracy vs number of basis vectors.
- **Data files**: `handwriting_training_set.txt`, `handwriting_training_set_labels.txt`, `handwriting_test_set.txt`, `handwriting_test_set_labels_for_Python.txt`. All are expected in the repository root and are loaded with `numpy.loadtxt`.
- **Why this structure**: the code separates examples by true digit, computes an SVD per class to obtain a per-digit subspace, then classifies by measuring distance from a test vector to each digit subspace.

**Key conventions & quirks (do not change unknowingly)**
- **Label encoding**: Digit `0` is encoded as `10` in the training labels. The code converts `10` → `0` when comparing predicted vs true labels. See `problem1.py` where it maps `true_label == 10` to `0`.
- **Subspace basis extraction**: `Vt = np.linalg.svd(...)[2]` is stored in `svd_results[digit]['Vt']`. The top-`k` basis is obtained as `V_k = svd_results[d]['Vt'][:k, :].T`. Keep this indexing pattern when using the stored `Vt`.
- **File paths**: paths are relative; run commands from the repository root (where the `.txt` files live).

**Data flow summary (useful anchor points)**
- Load: `np.loadtxt('handwriting_training_set.txt', delimiter=',')` and label files with no delimiter.
- Group: iterate digits 0–9 and collect examples into `digit_matrices[digit]` (note `0` uses indices where label == 10).
- SVD: `U, s, Vt = np.linalg.svd(digit_matrix, full_matrices=False)` saved in `svd_results`.
- Classify: for a test vector `x`, form `V_k`, compute projection `V_k @ (V_k.T @ x)` and distance `||x - projection||` — smallest distance = predicted digit.

**Dependencies & run steps**
- Required Python packages: `numpy`, `matplotlib`. There is no `requirements.txt` in the repo.
- Install (PowerShell):
```
python -m pip install --upgrade pip; python -m pip install numpy matplotlib
```
- Run the main script (from repo root):
```
python problem1.py
```

**Editing / extending guidance (concrete examples)**
- To evaluate more `k` values, edit `k_values = [5, 10, 15, 20]` in `problem1.py`. Accuracy is computed for each `k` and plotted.
- If adding an alternative classifier, create a new module (e.g., `svm_classifier.py`) and keep `problem1.py` as the experiment driver that imports and compares methods.
- When changing label handling, update both training and test label parsing sites — the code reads test labels from `handwriting_test_set_labels_for_Python.txt` and maps `10 -> 0` before accuracy checks.

**Safety checks / quick pitfalls**
- If `np.loadtxt` fails, verify file encoding and that `delimiter=','` is correct for the dataset files.
- Ensure the current working directory when running is the repository root — relative paths are used everywhere.
- Do not rename the data files unless you update all references in `problem1.py`.

**Where to look for examples**
- `problem1.py` — full implementation of the pipeline (data load → per-digit SVD → classification → plot). Use it as the canonical example for data layout, label quirks, and SVD indexing patterns.

**If you need to change tests / add CI**
- There are no tests or CI present. Add a `requirements.txt` and a simple `pytest`-based smoke test that runs `problem1.py` with a tiny sample if you add CI.

**If something is unclear**
- Ask specifically which behavior you want changed (e.g., label mapping, classifier replacement, or data format conversion). Provide a small code patch and a brief rationale so maintainers can accept changes quickly.

---
Please review these notes and tell me which areas you want expanded (example: add a sample unit test, add `requirements.txt`, or show how to refactor the classifier into reusable modules).
