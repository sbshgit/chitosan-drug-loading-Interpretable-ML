# Interpretable Machine Learning to Accelerate Formulation Development

## Predicting and Optimizing Drug Loading in Chitosan Nanoparticles

[![Python](https://img.shields.io/badge/Python-3.9.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the curated data, Python code, and analyses supporting the study **“Interpretable Machine Learning to Accelerate Formulation Development: Predicting and Optimizing Drug Loading in Chitosan Nanoparticles.”** The project develops an explainable machine-learning workflow to predict drug loading (DL) in chitosan-based nanoparticles and use model-derived insights to guide formulation development and laboratory testing.

---

## Graphical abstract

[**View the graphical abstract (Figure_1.tif)**](Figure_1.tif)

---

## Project overview

Achieving high drug loading in chitosan nanoparticles is challenging because loading depends on multiple, interrelated factors, including drug properties, polymer characteristics, and formulation parameters. This study combines literature-derived experimental data, regression modelling, explainability, and laboratory validation to investigate these relationships and support more informed formulation decisions.

### Dataset

The literature-curated dataset comprises:

- **141 experimental drug-loading measurements**
- **32 small-molecule drugs**
- **39 unique drug–chitosan pairs**
- **Four chitosan-based nanoparticle types:** conventional chitosan, carboxymethyl chitosan, trimethyl chitosan, and thiolated chitosan

The initial dataset included **15 input features** describing chitosan properties, formulation parameters, and molecular descriptors of the drug and formulation components. Following correlation-based feature selection, the final model used **14 features**.

## Methodological workflow

1. **Literature data curation** — Experimental drug-loading data and associated formulation and molecular information were collected from published studies.
2. **Preprocessing and feature selection** — Features were standardized. Spearman rank correlation and agglomerative hierarchical clustering were used to identify redundant variables. The modifier-to-chitosan ratio was removed because it was strongly correlated with the degree of substitution (Spearman's ρ = 0.96).
3. **Model benchmarking** — Ten regression algorithms were compared using nested cross-validation. The outer loop used 10-fold cross-validation, while the inner loop used 15-fold cross-validation and randomized hyperparameter search over 40 configurations.
4. **Model interpretation** — TreeSHAP was applied to the selected Natural Gradient Boosting (NGB) model to examine global feature importance and formulation-specific contributions. SHAP-derived attribution patterns were further explored using PCA and t-SNE.
5. **Laboratory validation** — Four model-guided formulations were prepared and tested. Experimental drug-loading measurements were compared with model predictions using Bland–Altman analysis.

### Machine-learning models

The following ten regression algorithms were evaluated:

| Model family | Algorithms |
| --- | --- |
| Linear | Multiple Linear Regression (MLR), Lasso, Partial Least Squares (PLS) |
| Kernel- and instance-based | Support Vector Regression (SVR), k-Nearest Neighbors (k-NN) |
| Tree-based | Decision Tree (DT), Random Forest (RF) |
| Boosting | Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), Natural Gradient Boosting (NGB) |

## Key findings

### Predictive performance

NGB showed the best overall performance among the evaluated models. The full 15-feature NGB model achieved a test mean absolute error (MAE) of **5.07**, a root mean squared error (RMSE) of **7.71**, and an R² of **0.74**. After removing the redundant feature, the final 14-feature NGB model retained comparable performance (test MAE **5.14**, RMSE **7.74**, and R² **0.73**) and was used for the subsequent interpretability and formulation analyses.

| NGB model | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: |
| 15 features | 5.07 | 7.71 | 0.74 |
| Final 14 features | **5.14** | **7.74** | **0.73** |

### Important predictors of drug loading

TreeSHAP identified the following as the most influential predictors, in descending order:

1. Drug-to-chitosan ratio (D/C ratio)
2. Particle size (PS)
3. Drug molecular weight (Drug_MW)
4. Crosslinker-to-chitosan ratio (Cross/C ratio)
5. Number of hydrogen-bond donors (NumHDonors)

The SHAP analysis associated higher predicted drug loading with higher drug-to-chitosan ratios, smaller particle sizes, higher drug molecular weights, lower crosslinker-to-chitosan ratios, and a greater number of hydrogen-bond donors. The SHAP–t-SNE analysis also highlighted high-loading regions characterized by medium-to-high D/C ratios, greater chitosan degree of deacetylation, moderately lipophilic drugs, at least three hydrogen-bond donors, and drug pKa values closer to physiological pH.

The effect of drug lipophilicity was polymer-dependent: the model-associated patterns differed between unmodified and thiolated chitosan formulations. These findings describe patterns learned from the curated dataset and should be interpreted within the scope of the studied formulation space.

### Model-guided laboratory validation

Four formulations were prepared using conventional chitosan or trimethyl chitosan (TMC), each with either betamethasone (lipophilic) or ciprofloxacin (hydrophilic). Each formulation was prepared in three independent batches (n = 3).

| Formulation | Experimental DL (%) (mean ± SD) | NGB-predicted DL (%) |
| --- | ---: | ---: |
| Chitosan–betamethasone | 34.93 ± 3.21 | 33.44 |
| TMC–betamethasone | 36.62 ± 2.30 | 36.28 |
| Chitosan–ciprofloxacin | 35.75 ± 1.48 | 34.18 |
| TMC–ciprofloxacin | 23.80 ± 1.84 | 23.55 |

Bland–Altman analysis showed a mean bias of **+0.91 percentage points**, with 95% limits of agreement from **−3.03 to +4.85 percentage points**. Both limits were within the predefined agreement margin of ±5 percentage points, supporting agreement between the model predictions and the experimental measurements for the tested formulations.

## Repository organization

```text
.
├── Box Plot/                         # Model error and performance visualizations
├── Feature selection/                # Feature correlation and selection analyses
├── Model-guided Laboratory Validation/ # Experimental validation analyses
├── Models Training/                  # Model training and benchmarking
├── SHAP/                             # Model explanation and SHAP analyses
├── data/                             # Curated data used in the study
├── Figure_1.tif                      # Graphical abstract
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

Clone the repository:

```bash
git clone https://github.com/sbshgit/chitosan-drug-loading-Interpretable-ML.git
cd chitosan-drug-loading-Interpretable-ML
```

The analyses were developed and tested using **Python 3.9.12**. The repository is organized by analysis stage; start with the relevant folder (`Models Training`, `Feature selection`, `SHAP`, or `Model-guided Laboratory Validation`) to locate the corresponding scripts and supporting files. Check the individual scripts and accompanying files for the required packages and execution details before running an analysis.

## Data and code availability

The curated dataset and Python implementation supporting this study are available in this repository. The code was developed and tested using Python 3.9.12.

## Study limitations

The dataset was compiled from published studies and focuses on small-molecule drugs formulated in chitosan-based nanoparticles. The resulting model may not generalize directly to macromolecular therapeutics, other nanocarrier classes, or formulation conditions outside the range represented in the training data. Model predictions are intended to support formulation prioritization and experimental planning; further testing is needed when applying the approach to new formulation spaces.

## How to cite

Please cite the article when using the data, code, or methods from this repository:

> Qaribnejad A, **Shiranirad S**, Movahedi S, Aftabi P, Varshochian R, Barzegar Z. **Interpretable machine learning to accelerate formulation development: Predicting and optimizing drug loading in chitosan nanoparticles.** *Journal of Pharmaceutical Sciences*. 2026:104628.

Amirsajad Qaribnejad and Saba Shiranirad contributed equally to this work.

## Authors and affiliations

**Amirsajad Qaribnejad**, **Saba Shiranirad**, **Sara Movahedi**, **Parnia Aftabi**, **Reyhaneh Varshochian**, and **Zeynab Barzegar**.

- Department of Pharmaceutics and Pharmaceutical Nanotechnology, School of Pharmacy, Shahid Beheshti University of Medical Sciences, Tehran, Iran.
- Department of Artificial Intelligence in Medicine, Advanced Technologies in Medicine, Iran University of Medical Sciences, Tehran, Iran.

## Acknowledgements

This study was funded by **Shahid Beheshti University of Medical Sciences**. We thank all co-authors and collaborators for their contributions to the computational and experimental work.

## License

The repository is distributed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

