# Interpretable Machine Learning to Accelerate Formulation Development

## Predicting and Optimizing Drug Loading in Chitosan Nanoparticles

[![Python](https://img.shields.io/badge/Python-3.9.12-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the curated dataset, Python implementation, and supporting analyses for the study **“Interpretable Machine Learning to Accelerate Formulation Development: Predicting and Optimizing Drug Loading in Chitosan Nanoparticles.”** The work presents an explainable machine-learning workflow for predicting drug loading (DL) in chitosan-based nanoparticles and supporting data-informed formulation development.

<p align="center">
  <img src="Graphical%20Abstract.jpg" alt="Graphical abstract of the study" width="850">
</p>

## Study overview

The literature-curated dataset comprises **141 experimental drug-loading measurements**, covering **32 small-molecule drugs**, **39 unique drug–chitosan pairs**, and four chitosan-based nanoparticle types: conventional chitosan, carboxymethyl chitosan, trimethyl chitosan, and thiolated chitosan.

Ten regression algorithms were benchmarked using **15 input features** describing polymer properties, formulation parameters, and molecular descriptors. Following correlation-based feature selection using Spearman rank correlation and agglomerative hierarchical clustering, a final **14-feature Natural Gradient Boosting (NGB)** model was developed and interpreted using TreeSHAP.

The most influential predictors of drug loading were:

1. Drug-to-chitosan ratio (D/C ratio)
2. Particle size (PS)
3. Drug molecular weight (Drug_MW)
4. Crosslinker-to-chitosan ratio (Cross/C ratio)
5. Number of hydrogen-bond donors (NumHDonors)

## Repository organization

```text
.
├── Models Training/                    # Training and benchmarking of 10 ML models using 15 features
├── Box Plot/                           # Box-plot visualizations of model performance and prediction errors
├── Feature selection/                  # Spearman correlation and agglomerative clustering; selection of 14 features and final NGB model training
├── SHAP/                               # TreeSHAP analyses: global feature importance, beeswarm, decision paths, force plots, and t-SNE
├── Model-guided Laboratory Validation/ # Analyses of the experimental validation of model-guided formulations
├── data/                               # Curated data used in the study
├── LICENSE
└── README.md
```

## Software environment

The code and analyses use the following Python environment and library versions:

| Software / library | Version |
| :----------------- | :-----: |
| Python             | 3.9.12  |
| NumPy              | 1.26.4  |
| pandas             |  2.3.3  |
| Matplotlib         |  3.9.0  |
| Seaborn            | 0.13.2  |
| scikit-learn       |  1.6.1  |
| XGBoost            |  2.1.4  |
| LightGBM           |  4.3.0  |
| NGBoost            |  0.5.1  |
| SHAP               | 0.49.1  |
| openpyxl           |  3.1.2  |

## How to cite

Please cite the following article when using the data, code, or methods from this repository:

> Qaribnejad A, **Shiranirad S**, Movahedi S, Aftabi P, Varshochian R, Barzegar Z. **Interpretable machine learning to accelerate formulation development: Predicting and optimizing drug loading in chitosan nanoparticles.** _Journal of Pharmaceutical Sciences_. 2026:104628.

## Authors and affiliations

**Amirsajad Qaribnejad**, **Saba Shiranirad**, **Sara Movahedi**, **Parnia Aftabi**, **Reyhaneh Varshochian**, and **Zeynab Barzegar**.

- Department of Pharmaceutics and Pharmaceutical Nanotechnology, School of Pharmacy, Shahid Beheshti University of Medical Sciences, Tehran, Iran.
- Department of Artificial Intelligence in Medicine, Advanced Technologies in Medicine, Iran University of Medical Sciences, Tehran, Iran.

## License

This repository is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
