# Interpretable Machine Learning for Drug Loading in Chitosan Nanoparticles

This repository contains the curated dataset and Python code accompanying our study on interpretable machine-learning prediction and optimization of drug loading in chitosan nanoparticles.

![Graphical Abstract](Graphical%20Abstract.jpg)

## Repository organization

```text
.
├── Models Training/                      # Training and benchmarking of 10 ML models using 15 features
├── Box Plot/                              # Box plots visualizing model performance and prediction errors
├── Feature selection/                     # Spearman correlation and agglomerative clustering; selection of 14 features and final NGB model training
├── SHAP/                                   # TreeSHAP interpretation: global feature importance, beeswarm, decision paths, force plots, and t-SNE
├── Model-guided Laboratory Validation/    # Analyses of the experimental validation of model-guided formulations
├── data/                                   # Curated data used in the study
├── LICENSE
└── README.md
```

## Citation

Please cite our work when using this repository:

Qaribnejad A, Shiranirad S, Movahedi S, Aftabi P, Varshochian R, Barzegar Z. **Interpretable machine learning to accelerate formulation development: Predicting and optimizing drug loading in chitosan nanoparticles.** *Journal of Pharmaceutical Sciences*. 2026:104628.
