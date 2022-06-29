from joblib import load
import pandas as pd
from pathlib import Path

datapath = Path('/media/sf/easystore/kaggle_data/tabular_playgrounds/sep2021/')

X_np = load(datapath/'X_SimpleImputed_StandardScaled_PolyDeg2wInteract_np.joblib')
print('creating dataframe')
X = pd.DataFrame(X_np, columns=[str(x) for x in range(X_np.shape[1])])
print("deleting X_np")
del X_np
print("saving as feather")
X.to_feather(datapath/'X_SimpleImputed_StandardScaled_PolyDeg2wInteract.feather')
