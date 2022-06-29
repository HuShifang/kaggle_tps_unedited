# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

# general ML tooling
# from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
# from sklearn.metrics import roc_auc_score
# import wandb
# from wandb.xgboost import wandb_callback
# import timm
from pathlib import Path
# import os
# import math
import seaborn as sns
from datetime import datetime

# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from joblib import dump, load
# from sklearn.impute import SimpleImputer, KNNImputer
# feature engineering tools
# from sklearn.feature_selection import mutual_info_regression
# import featuretools as ft
# import missingno as msno
from sklearn.decomposition import IncrementalPCA


datapath = Path('/media/sf/easystore/kaggle_data/tabular_playgrounds/sep2021/')

print(f"starting at {datetime.now().strftime('%Y%m%d%H%M%S')}")
X = load(datapath/'X_SimpleImputed_StandardScaled_PolyDeg2wInteract_np.joblib')
pca = IncrementalPCA(n_components=6000, copy=False) # 'mle' not accepted for IncrementalPCA
X_pca = pca.fit_transform(X)
dump(X_pca, datapath/'X_SimpleImputed_StandardScaled_PolyDeg2wInteract_pca-mle_20210907.joblib')

pca_ratios = pca.explained_variance_ratio_
dump(pca_ratios, datapath/'pca-full_ratios_20210906.joblib')

sns_plot = sns.lineplot(data=np.cumsum(pca.explained_variance_ratio_)) 
fig = sns_plot.get_figure()
fig.savefig(datapath/'pca_elbow_plot_20210906.png')
print(f"done at {datetime.now().strftime('%Y%m%d%H%M%S')}")
