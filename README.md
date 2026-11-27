# Interpretable Machine Learning to Accelerate Formulation Development

## Predicting and Optimizing Drug Loading in Chitosan Nanoparticles

[![Python](https://img.shields.io/badge/Python-3.9.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the curated data, Python code, and analyses supporting the study **“Interpretable Machine Learning to Accelerate Formulation Development: Predicting and Optimizing Drug Loading in Chitosan Nanoparticles.”** The project develops an explainable machine-learning workflow to predict drug loading (DL) in chitosan-based nanoparticles and use model-derived insights to guide formulation development and laboratory testing.

![Graphical Abstract](Graphical%20Abstract.jpg)

## Project overview

Achieving high drug loading in chitosan nanoparticles is challenging because loading depends on multiple, interrelated factors, including drug properties, polymer characteristics, and formulation parameters. This study combines literature-derived experimental data, regression modelling, explainability, and laboratory validation to investigate these relationships and support more informed formulation decisions.

### Dataset

The literature-curated dataset comprises:

- **141 experimental drug-loading measurements**
- **32 small-molecule drugs**
- **39 unique drug–chitosan pairs**
- **Four chitosan-based nanoparticle types:** conventional chitosan, carboxymethyl chitosan, trimethyl chitosan, and thiolated chitosan

The initial dataset included **15 input features** describing chitosan properties, formulation parameters, and molecular descriptors of the drug and formulation components. Following correlation-based feature selection, the final model used **14 features**.

### Machine-learning models

The following ten regression algorithms were evaluated:

Multiple Linear Regression (MLR), Lasso, Partial Least Squares (PLS), Support Vector Regression (SVR), k-Nearest Neighbors (k-NN), Decision Tree (DT), Random Forest (RF), Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), Natural Gradient Boosting (NGB)

### Important predictors of drug loading

TreeSHAP identified the following as the most influential predictors, in descending order:

1. Drug-to-chitosan ratio (D/C ratio)
2. Particle size (PS)
3. Drug molecular weight (Drug_MW)
4. Crosslinker-to-chitosan ratio (Cross/C ratio)
5. Number of hydrogen-bond donors (NumHDonors)

## Repository organization

```text
.
├── Models Training/                       # Training of 10 Machine Learning models using 15 features
├── Box Plot/                              # Box plots visualizing model performance and prediction errors
├── Feature selection/                     # Spearman correlation and agglomerative clustering; selection of 14 features and final NGB model training
├── SHAP/                                  # TreeSHAP interpretation: global feature importance, beeswarm, decision paths, force plots, and t-SNE
├── Model-guided Laboratory Validation/    # Analyses of the experimental validation of model-guided formulations
├── data/                                  # Curated data used in the study
├── LICENSE
└── README.md
```

## How to cite

Please cite the article when using the data, code, or methods from this repository:

> Qaribnejad A, Shiranirad S, Movahedi S, Aftabi P, Varshochian R, Barzegar Z. **Interpretable machine learning to accelerate formulation development: Predicting and optimizing drug loading in chitosan nanoparticles.** _Journal of Pharmaceutical Sciences_. 2026:104628.

## Authors and affiliations

**Amirsajad Qaribnejad**, **Saba Shiranirad**, **Sara Movahedi**, **Parnia Aftabi**, **Reyhaneh Varshochian**, and **Zeynab Barzegar**.

- Department of Pharmaceutics and Pharmaceutical Nanotechnology, School of Pharmacy, Shahid Beheshti University of Medical Sciences, Tehran, Iran.
- Department of Artificial Intelligence in Medicine, Advanced Technologies in Medicine, Iran University of Medical Sciences, Tehran, Iran.

## License

The repository is distributed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.
