# FSL vs. Classical ML on Reduced Tabular Data — Code and Results

Code and result files supporting the paper *"Comparative Analysis of Classical Machine Learning and Few-Shot Learning Methods Using Prototype Reduction Techniques on tabular data"* (Ramos-Aguilar, Olvera-López, Ramos-Aguilar).

This repository is referenced from the paper's **Data Availability** statement.

## Repository structure

```
reduction/          Prototype/instance reduction methods and synthetic-data generation (paper Sections "Prototype Selection Methods" and "Synthetic Data (SD)")
eval_synthetic/      Episodic evaluation of the 17 models on the 6 synthetic decision-boundary shapes (paper Section "Ranking across datasets" / synthetic results)
eval_realworld/      Episodic evaluation of the 17 models on the 12 real-world UCI/sklearn tabular datasets (paper Section "Results", real-world subsection)
```

### `reduction/`
- `genericData.py` — generates the six synthetic 2D shapes (Circle, XOR, Moons, Radial, Complex, Sinusoidal), clean and noisy versions.
- `reductionMethods.py` — implements the six prototype reduction methods used throughout the paper (CNN, ENN, RENN, Tomek Links, DROP3, PSC).
- `generated_datasets/` — raw synthetic datasets before reduction.
- `reduced_csv/` — synthetic datasets after applying each reduction method.
- `Figure_1.png` — reference figure of the reduction methods' effect.

### `eval_synthetic/`
- `evalGeneric.py` — episodic ($N$-way $K$-shot) evaluation harness for the 5 classical classifiers, 5 classical-on-ProtoNet-embedding (ProtoEmb) variants, and 7 FSL methods, applied to the synthetic shapes.
- `reduced_csv/` — input data (per shape, per reduction method).
- `reduction_report.csv` — summary of the reduction step for the synthetic data.
- `results/` — per-shape outputs (`01_circle` … `06_xor`), each containing the clean/noisy/reduced CSVs used as input and the corresponding `fewshot_results*.csv` evaluation output.

### `eval_realworld/`
- `evalGeneric.py` — same evaluation harness, adapted for real-world tabular data (adds robust CSV separator/target-column auto-detection and a Windows-multiprocessing fallback for `predict()`, since real-world files are less uniform than the synthetic ones).
- `recompute_paper_tables.py` — aggregates the raw episodic results into the accuracy/F1/Wilcoxon tables reported in the paper.
- `reduced_csv/` — the 12 real-world datasets (UCI Machine Learning Repository / sklearn / seaborn / GitHub sources, see the paper's dataset table) after each reduction method.
- `FINAL_acc_table.csv`, `FINAL_f1_table.csv`, `FINAL_wilcoxon.csv` — final aggregated tables reported in the paper (accuracy, macro-F1, and Bonferroni-corrected Wilcoxon significance tests).

## Requirements

```bash
pip install -r requirements.txt
```

Developed with Python 3.x, `scikit-learn`, `tensorflow`, `pandas`, `numpy`, `scipy`, and `matplotlib`.

## Reproducing the pipeline

1. **Generate and reduce data** — run `reduction/genericData.py` to generate the synthetic shapes, then `reduction/reductionMethods.py` to produce the reduced CSVs (or use the `reduced_csv/` folders already provided).
2. **Evaluate on synthetic data** — run `eval_synthetic/evalGeneric.py` against `eval_synthetic/reduced_csv/`.
3. **Evaluate on real-world data** — run `eval_realworld/evalGeneric.py` against `eval_realworld/reduced_csv/`, then `eval_realworld/recompute_paper_tables.py` to reproduce the final tables (`FINAL_acc_table.csv`, `FINAL_f1_table.csv`, `FINAL_wilcoxon.csv`).

Each script's random seeds ($s \in \{7, 21, 42, 123, 456\}$) and episodic protocol match the methodology described in the paper.

## Citation

If you use this code or the accompanying results, please cite the paper (details to be added upon publication).
