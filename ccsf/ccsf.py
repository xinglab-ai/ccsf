# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:40:56 2022
@author: Md Tauhidul Islam, Postdoc, Xing Lab,
Department of Radiation Oncology, Stanford University
 
"""
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy import stats
import phate
import novosparc
from sklearn import metrics as mc
from scipy.spatial.distance import cdist
from ccsf.class_discriminative_opt import ClassDiscriminative_OPT
from ccsf.group_centroid_opt import groupCentroidOPT
from sklearn.preprocessing import normalize
import scanpy as sc

def CCSF_analysis(data, metric='correlation', n_clusters=32, num_comp=31, random_state=0):
    """
     Cell-cell similarity-driven framework 
    
    Parameters:
        
        data: High dimensional gene expression data in tabular
        format. The rows denote the cells and columns denote the genes.
        
        metric: Similarity metric denoting the metric to compute the interaction among cells, string, optional, default:'correlation'
        Other choices are ['euclidean', 'chebyshev','correlation','cityblock','sqeuclidean']
        
        n_clusters: int, optional, default: 32.
        number of groups in the data
        
        num_comp: int, optional, default: n_clusters-1
        Number of CCIF components to be extracted
        
        random_state: random state parameter for initialization of the method
    """
    # First optimization (group centroid optimization)
    grpCenterOPT=groupCentroidOPT(X=data,k=n_clusters,maxiter=100, metric=metric)
    clusterCenter=grpCenterOPT.centres
    labeldata=grpCenterOPT.Xtocentre
    Kdis = pairwise_distances(data, clusterCenter, metric=metric)
    # Compute cell-centroid similarity
    Kdis = normalize(Kdis, axis=1, norm='l1')
    Kdis=1-Kdis
    # Second optimization (class-discriminative optimization)
    clf = ClassDiscriminative_OPT(n_components=num_comp)
    clf.fit(Kdis, labeldata)
    ccifOut = clf.transform(Kdis)
    return ccifOut


class CCSF:
    """
    CCSF operator which performs cell-cell similarity-driven exploration of the gene expression
    datasets
    
    Ref: Leveraging cell-cell similarity for high-performance spatial and temporal cellular
    mappings from gene expression data, Accepted in Cell Patterns.
    ...

    Attributes
    ----------
    metric : str
        ['euclidean', 'chebyshev','correlation','cityblock','sqeuclidean']

    Methods
    -------
    metric_learning(self, data, labeldata):
        Learn the similarity metric that provides maximum separation of data
        classes in low dimension
    fit_transform(self, data):
        Return the low dimensional representation of CCIF
    """

    def __init__(self, metric='correlation',
                 n_clusters=8, num_comp=8, random_state=0):
        """
        Initialization of CCSF method
        """

        if metric in ['euclidean', 'chebyshev','correlation','cityblock','sqeuclidean']:
            self.metric = metric
        else:
            print('Unknown metric. use "correlation" as default.')
            self.metric = "correlation"

        self.random_state = random_state
        self.n_clusters = n_clusters
        self.num_comp = num_comp

    def metric_learning(self, data, lowdim_num=2, resolution=1.0, verbose=False):
        """
        Learns the similarity metric that results in best clustering of the 
        gene expression data in lower dimension 
        
        Parameters:
        
        data: High dimensional gene expression data in tabular
        format. The rows are the cells and columns are the genes.
        
        lowdim_num:int, optional, default: 2 
            The number of dimension, where the class separation is examined            
        resolution: float, optional, default: 1.0
            The resolution parameter in Leiden algorithm 
        verbose: boolean, optional, default: false
            if true, the code prints the cluster quality values for each of the 
            distance metric considered
        """

        metrics = ['chebyshev','correlation','cityblock','sqeuclidean']
        lenMetrics=len(metrics)
        lenQualityIndices=3 # Change if you use different number of cluster 
        # quality indice. We used 3 indices.
        matClusterQuality=np.zeros((lenMetrics,lenQualityIndices))
        matNumClass=np.zeros((lenMetrics,1))
        i=0
        max_metric = ''

        for metric in metrics:

            kmedoids = KMedoids(n_clusters=lowdim_num, metric=metric,
                                random_state=self.random_state).fit(data)
            clusterCenter = kmedoids.cluster_centers_
            dist = pairwise_distances(data, clusterCenter, metric=metric)

            # Find number of cell types/data classes
            adata = convertToAnnData(data)
            # Perform Leiden clustering
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata, resolution=resolution)
            # Access the cluster assignments
            labeldata_estimate = adata.obs['leiden']
            numClass = len(np.unique(labeldata_estimate)) 

            silVal=mc.silhouette_score(data, labeldata_estimate, metric=metric)
            dbVal=mc.davies_bouldin_score(data, labeldata_estimate)
            chVal=mc.calinski_harabasz_score(data, labeldata_estimate)
            x = np.array([silVal, -dbVal, chVal])
            matClusterQuality[i]=x
            matNumClass[i]=numClass
            i=i+1            
            
            if verbose:
                 print(metric + ': ' + str(x))
                 
        # Finding out the metric with the best cluster quality indices. For this,
        # row by row comparison is necessary in matClusterQuality matrix.  
        compareValMat=-5000*np.ones(lenMetrics) # This matrix saves how many metrics
        # is worse than each of the metric. The matrix with most value in this matrix
        # is the best metric. 
        for ii in range(lenMetrics):
            saveVal=0
            for jj in range(lenMetrics):
                 whetherGood=matClusterQuality[ii]>matClusterQuality[jj] # comparing each
                 # metric with all others
                 whetherGood = np.multiply(whetherGood, 1) # converting to numbers
                 saveVal=saveVal+np.sum(whetherGood) # how many times is a metric is better
                 
            compareValMat[ii]=saveVal
        
        # Next, we find if two metric is better than the same number of other metrics
        max_val = max(compareValMat)
        i = 0
        ind = []
        for val in compareValMat:
            if val == max_val:
                ind.append(i)
            i = i + 1            

        if len(ind)>1:  # if the two metric is better than the same number of other metrics          
            whetherGood=matClusterQuality[ind[0]]>matClusterQuality[ind[1]]
            whetherGood = np.multiply(whetherGood, 1) # converting to numbers
            sumGood=np.sum(whetherGood)
            if sumGood>1:
                max_index_col=np.asarray (ind[0])
            else:
                max_index_col=np.asarray(ind[1])
        else:
            max_index_col=np.asarray(ind)   
            
        self.metric = metrics[max_index_col.item()]
        max_metric=metrics[max_index_col.item()]
        numClassSelected=int(np.squeeze(matNumClass[max_index_col.item()]))
        return max_metric,numClassSelected

    def cMap(self, data):
        """
        CCSF-UMAP (cMap) visualizes the high dimensional gene
        expression data with different number of CCSF components
        
        Parameters:
        
        data: High dimensional gene expression data in tabular
        format. The rows are the cells and columns are the genes.
        """
        assert 'data' in data, ' "data" must be feeded in the input.'
        data_transform = data['data']

        nummaxCCSFcomponent=np.max(self.num_comp)
        
        ccifOut = CCSF_analysis(data_transform, n_clusters=self.n_clusters,
                              num_comp=nummaxCCSFcomponent, metric=self.metric)

        lenX=self.num_comp.shape[0]
        embeddingCollection=[] # Collection of embeddings for all the CCSF components
        for ii in range(lenX):
            numC=self.num_comp[ii]
            ccifOutX=ccifOut[:,0:numC]
            ccifOutzc = stats.zscore(ccifOutX, axis=0, ddof=1)# Z-score of the CCSF 
            # components
            # Apply UMAP
            embedding = umap.UMAP(n_neighbors=30, min_dist=0.3,init="random", 
                                  n_epochs=200).fit_transform(ccifOutzc)
            embeddingCollection.append(embedding)
            
        return embeddingCollection
    
    
    def cPHATE(self, data):
        """
        CCSF-PHATE (cPHATE) visualizes the high dimensional gene expression 
        data with different number of CCSF components for cellular trajectory
        analysis
        
        Parameters:
        
        data: High dimensional gene expression data in tabular
        format. The rows are the cells and columns are the genes.
        """
        assert 'data' in data, ' "data" must be feeded in the input.'
        data_transform = data['data']
        
        nummaxCCSFcomponent=np.max(self.num_comp)
        
        ccifOut = CCSF_analysis(data_transform, n_clusters=self.n_clusters,
                              num_comp=nummaxCCSFcomponent, metric=self.metric)

        lenX=self.num_comp.shape[0]
        embeddingCollection=[] # Collection of embeddings for all the CCSF components
        # Apply PHATE
        for ii in range(lenX):
            numC=self.num_comp[ii]
            ccifOutX=ccifOut[:,0:numC]
            ccifOutzc = stats.zscore(ccifOutX, axis=0, ddof=1) # Z-score of the CCIF 
            # components
            phate_op = phate.PHATE()
            embedding = phate_op.fit_transform(ccifOutzc)
            embeddingCollection.append(embedding)
            
        return embeddingCollection



    def cSpaRc(self, data, locations):
        """
        CCSF-novoSpaRc (cSpaRc) reconstructs the spatial maps of genes from 
        scRNA-seq data
        
        Parameters:
        
        data: High dimensional scRNA-seq data in tabular
        format. The rows are the cells and columns are the genes.
        """
        assert 'data' in data, ' "data" must be feeded in the input.'
        data_transform = data['data']

        num_cells = data_transform.shape[0]
        num_locations = locations.shape[0]

        # Find the CCSF components
        corrData = CCSF_analysis(
            data_transform, n_clusters=self.n_clusters, num_comp=self.num_comp, metric=self.metric)

        # Use CCIF components as the marker genes in the spatial reconstruction by novoSpaRc
        cost_expression, cost_locations = novosparc.rc.setup_for_OT_reconstruction(data_transform[:, :],
                                                                                   locations,
                                                                                   num_neighbors_source=5,
                                                                                   num_neighbors_target=5)

        cost_marker_genes = cdist(corrData/np.amax(corrData),
                                  corrData/np.amax(corrData))

        # Distributions at target and source spaces
        p_locations, p_expression = novosparc.rc.create_space_distributions(
            num_locations, num_cells)

        alpha_linear = 0.5
        gw = novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
                                                                       alpha_linear, p_expression, p_locations,
                                                                       'square_loss', epsilon=5e-4, verbose=True)

        sdge_DGE_CORR = np.dot(data_transform.T, gw)
        return sdge_DGE_CORR

def convertToAnnData(data):
    # Create pseudo cell names
    cell_names = ['Cell_' + str(i) for i in range(1, data.shape[0]+1)]
    # Create pseudo gene names
    gene_names = ['Gene_' + str(i) for i in range(1, data.shape[1]+1)]
    # Create a pandas DataFrame
    df = pd.DataFrame(data, index=cell_names, columns=gene_names)
    adataMy=sc.AnnData(df)
    return adataMy    

