import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biclust import biclust, biclusternumber
from MSA import ausActiv, ausActivDesc  # Assuming you have the MSA package installed

# Set seed
np.random.seed(1234)

# Bi-clustering solution for Australian vacation activities
ausact_bic = biclust(x=ausActiv, method='BCrepBimax', minc=2, minr=50, number=100, maxc=100)

# Obtain the segment memberships from the bi-clustering solution
bcn = biclusternumber(ausact_bic)
cl12 = np.full(nrow(ausActiv), np.nan)
for k in range(len(bcn)):
    cl12[bcn[k].Rows] = k + 1

print(pd.Series(cl12).value_counts(dropna=False))

cl12_3 = pd.Series(~pd.Series(cl12).isna() & pd.Series(cl12 == 3), dtype='category')
cl12_3.cat.categories = ['Not Segment 3', 'Segment 3']

# 9.3 Price
boxplot_data = ausActivDesc.groupby(cl12_3)['spendpppd'].apply(np.log1p)
boxplot_data.index = boxplot_data.index.astype(str)  # Convert categories to strings for plotting
boxplot_data.plot(kind='box', notch=True, vert=False, widths=0.7)
plt.xlabel('log(AUD per person per day)')
plt.show()

# 9.4 Place
from flexclust import propBarchart
propBarchart(ausActivDesc, g=cl12_3,
             which=[col for col in ausActivDesc.columns if col.startswith('book')],
             layout=(1, 1), xlab='percent', xlim=(-2, 102))
plt.show()

# 9.5 Promotion
propBarchart(ausActivDesc, g=cl12_3,
             which=[col for col in ausActivDesc.columns if col.startswith('info')],
             layout=(1, 1), xlab='percent', xlim=(-2, 102))
plt.show()

# Mosaicplot
plt.figure()
plt.xticks(rotation=90)
sns.heatmap(pd.crosstab(index=cl12_3, columns=ausActivDesc['TV.channel'], margins=True, margins_name='Total'),
            annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('TV Channel')
plt.ylabel('Segment')
plt.title('Mosaicplot')
plt.show()
