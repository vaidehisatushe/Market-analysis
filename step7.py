import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text

# Load the data
from flexclust import vacmot
from flexclust.utils import relabel
from flexclust.cluster import StepC

# 7.2 Using visualisation to describe market segments

# Set the seed
np.random.seed(1234)

# Create a StepC clustering model
vacmot_k38 = StepC(k=range(3, 9), method="neuralgas", nrep=20, save_data=True, verbose=False)
vacmot_k38.fit(vacmot)
vacmot_k38 = relabel(vacmot_k38)
vacmot_k6 = vacmot_k38['6']

C6 = vacmot_k6.get_clusters()
print(pd.Series(C6).value_counts())

# Add C6 to the original dataset
vacmotdesc['C6'] = C6.astype('category')

# 7.2.1 Visualising nominal and ordinal descriptor variables

C6_Gender = pd.crosstab(index=vacmotdesc['C6'], columns=vacmotdesc['Gender'])
print(C6_Gender)

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
sns.barplot(x=C6_Gender.columns, y=C6_Gender.loc[1], color="grey20", label="Male")
sns.barplot(x=C6_Gender.columns, y=C6_Gender.loc[2], color="grey80", label="Female")
plt.ylabel("Number of segment members")
plt.legend()

plt.subplot(2, 1, 2)
sns.mosaicplot(C6_Gender.unstack(), title="")
plt.show()


