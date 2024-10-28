#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np 
from gemelli.preprocessing import rclr_transformation
from fancyimpute import IterativeSVD
from skbio.stats.composition import clr, multiplicative_replacement
from gemelli.preprocessing import matrix_rclr
from gemelli.matrix_completion import MatrixCompletion


# In[18]:


def multiplicative_clr(ft):
    ft = ft.copy().to_dataframe().transpose()
    ft_array = np.array(ft)  
    # Replace zeros using multiplicative replacement
    metagenomic_counts = multiplicative_replacement(ft_array)

    # Apply CLR transformation
    clr_transformed_data = clr(metagenomic_counts)

    ft = pd.DataFrame(clr_transformed_data,index=ft.index, columns=ft.columns)
    
    return ft


# In[8]:


def fancy_iterativesvd(ft, rank, max_iter):
    rclr_table = rclr_transformation(ft).to_dataframe().transpose()
    rclr_array = rclr_table.to_numpy()
    svd_imputer = IterativeSVD(
        rank=rank,             # Adjust rank based on your data's dimensionality
        max_iters=max_iter,       # Maximum number of iterations
        convergence_threshold=1e-5  # Convergence tolerance
    )

    # Fit and transform your matrices
    rclr_svd = svd_imputer.fit_transform(rclr_array)
    ft = ft.to_dataframe().transpose()
    ft = pd.DataFrame(rclr_svd, ft.index, ft.columns)
    
    return ft


# In[10]:


def matrix_completion(ft):
    rclr_table = matrix_rclr(ft.matrix_data.toarray().T)
    opt = MatrixCompletion(n_components=10,
                       max_iterations=5).fit(rclr_table)
    # get completed matrix for centering
    X = opt.sample_weights @ opt.s @ opt.feature_weights.T
    
    # # center again around zero after completion
    # X = X - X.mean(axis=0)
    # X = X - X.mean(axis=1).reshape(-1, 1)
    
    ft = ft.to_dataframe().transpose()
    ft = pd.DataFrame(opt.solution, ft.index, ft.columns)
    
    return ft


# In[ ]:




