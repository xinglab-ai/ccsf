# ccsf
Leveraging cell-cell similarity for high-performance spatial and temporal cellular mappings from gene expression data (Cell Patterns, 2023)

## CCSF is a replacement of PCA for gene expression and other tabular data analysis
CCSF is a cell-cell similarity-driven framework of genomic data analysis for high-fidelity dimensionality reduction, clustering, visualization, and spatial and temporal cellular mappings. 
The approach exploits the similarity features of the cells for the discovery of discriminative patterns in the data. 
For a wide variety of datasets, the proposed approach drastically improves the accuracies of visualization and spatial and temporal mapping analyses as compared to PCA and state-of-the-art techniques. 
Computationally, the method is about 15 times faster than the existing ones and thus provides an urgently needed technique  for reliable and efficient analysis of genomic data.

## Sample data

To run the example codes below, you will need to download data files from [here](https://drive.google.com/drive/u/0/folders/1YNHD7CJeiCioJ21-yWrK8JpZlBYHzfe4).

## Example codes

### Example 1 - Learning the best metric and number of data classes

```python
# Import all the necessary Python packages
from ccsf import CCSF

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from scipy import stats
import numpy as np
import novosparc
import random

rngState=0 # For reproducibility

# Load TCGA data
data = pd.read_csv('tcga_data.csv', header=None,
                   delim_whitespace=False)
label = pd.read_csv('tcga_label.csv', header=None,
                         delim_whitespace=False).to_numpy()

data = data.to_numpy()
n = 2000 # Select 2000 points randomly from the data for learning the distance metric.
manifolder = CCSF(random_state=rngState)
random.seed(a=rngState) 
sampleix = random.sample(range(data.shape[0]), int(n) )
dataSampled = data[sampleix]
numInitCls = int(np.floor(data.shape[0]/500))
dataSampled = stats.zscore(dataSampled, axis=0, ddof=1)
metric,numClass = manifolder.metric_learning(dataSampled, verbose=False)
print('metric for cMAP is:', metric)
# # You are supposed to see 'correlation' as the result.
```

### Example 2 - cMAP

```python
# Import all the necessary Python packages
from ccsf import CCSF

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from scipy import stats
import numpy as np
import novosparc
import random

rngState=0 # For reproducibility

# Load TCGA data
data = pd.read_csv('tcga_data.csv', header=None,
                   delim_whitespace=False)
label = pd.read_csv('tcga_label.csv', header=None,
                         delim_whitespace=False).to_numpy()

data= data.to_numpy()
feed_data = {'data': data}
manifolder = CCSF(random_state=rngState)
metric, numClass = manifolder.metric_learning(dataSampled, verbose=False)
num_comp= np.array([numClass-1]) # The number of CDM components to be used
# to compute the cMAP
manifolder = CCSF(n_clusters=numClass, num_comp=num_comp,metric=metric)
embedding_CMAP = manifolder.cMap(data=feed_data)
embedding = embedding_CMAP[0]
plt.figure()
plt.title('cMAP visualization ')
plt.scatter(embedding[:,0],embedding[:,1],c=label.T,s=0.5)
plt.xlabel("cMAP1")
plt.ylabel("cMAP2")
plt.show()
```

### Example 3 - Temporal mapping by CCSF

```python
# Import all the necessary Python packages
from ccsf import CCSF

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from scipy import stats
import numpy as np
import novosparc
import random

rngState=0 # For reproducibility

data = pd.read_csv('organoidData.csv', header=None,
                    delim_whitespace=False)
label = pd.read_csv('organoidDataLabel.csv', header=None,
                          delim_whitespace=False).to_numpy()
data = data.to_numpy()

# Learning the metric for cPHATE
n = 2000
manifolder = CCSF(random_state=rngState)
random.seed(a = rngState) 
sampleix = random.sample( range( data.shape[0] ), int(n) )
dataSampled=data[sampleix]
numInitCls = int(np.floor(data.shape[0]/500))
dataSampled = stats.zscore(dataSampled, axis=0, ddof=1)
metric,numClass = manifolder.metric_learning(dataSampled, verbose=False)
print('metric for cPHATE is:', metric)

numClass = 33 # Fixed for cellular trajectory analysis tasks
data = stats.zscore(data, axis=0, ddof=1)
num_comp = np.array([numClass-1]) # Number of CCSF components
manifolder = CCSF(n_clusters=numClass, num_comp=num_comp,metric=metric,random_state=rngState)

# cPHATE
feed_data = {'data': data}
embedding_CPHATE = manifolder.cPHATE(data=feed_data)

embedding = embedding_CPHATE[0]
plt.figure()
plt.title('cPHATE for 32 CCIF components ')
plt.scatter(embedding[:,0],embedding[:,1],c=label.T,s=0.5)
plt.xlabel("cPHATE1")
plt.ylabel("cPHATE2")
plt.show() 
```

### Example 4 - Spatial mapping by CCSF

```python
# Import all the necessary Python packages
from ccsf import CCSF

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from scipy import stats
import numpy as np
import novosparc
import random

rngState=0 # For reproducibility

# Read the BDTNP database
print ('Loading data ... ', end='', flush=True)
gene_names = np.genfromtxt('dge.txt', usecols=range(84),
                          dtype='str', max_rows=1)
dge = np.loadtxt('dge.txt', usecols=range(84), skiprows=1)
# Optional: downsample number of cells
cells_selected, dge = novosparc.pp.subsample_dge(dge, 3039, 3040)
num_cells = dge.shape[0]    
data=dge
feed_data = {'data': data}

# Learning the metric for cSPARC
n=2000
random.seed(a=rngState)
sampleix = random.sample( range( data.shape[0] ), int(n) )
dataSampled=data[sampleix]
manifolder = CCSF(random_state=rngState)
dataSampled = stats.zscore(dataSampled, axis=0, ddof=1)
numInitCls=int(np.floor(data.shape[0]/500))
metric,numClass = manifolder.metric_learning(dataSampled, verbose=False)
print('metric for CSPARC is:', metric)

print ('Reading the target space ... ', end='', flush=True)    
    # Read and use the bdtnp geometry
locations = np.loadtxt('geometry.txt', usecols=range(3), skiprows=1)
locations = locations[:, [0, 2]]
locations = locations[cells_selected, :] # downsample to the cells selected above

# Compute the spatial maps of the genes
manifolder = CCSF(n_clusters=numClass,num_comp=numClass-1,metric=metric)        
embedding_CSPARC = manifolder.cSpaRc(data=feed_data,locations=locations)


gene='ftz'
#gene='sna' Try other gene in place of ftz

d=embedding_CSPARC[np.argwhere(gene_names == gene), :].flatten()
plt.figure()
plt.title('cSpaRc spatial reconstruction of ftz gene')
plt.scatter(locations[:,0],locations[:,1],c=d.T)
plt.show()
```








