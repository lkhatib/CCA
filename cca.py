#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import SparsePCA
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm


# In[1]:


class CCAAnalysis:
    def __init__(self, n_components, output_path):
        self.n_comp = n_components
        self.output_path = output_path
        self.cca = None
        self.X1_c = None
        self.X2_c = None

    def perform_cca(self, df1, df2):
        self.cca = CCA(n_components=self.n_comp)
        self.cca.fit(df1, df2)
        self.X1_c, self.X2_c = self.cca.transform(df1, df2)

    def scree_plot(self):
        comp_corr = [np.corrcoef(self.X1_c[:, i], self.X2_c[:, i])[1][0] for i in range(self.n_comp)]

        cc_list = [f'CC{i+1}' for i in range(self.n_comp)]
        plt.bar(cc_list, comp_corr, color='lightgrey', width=0.8, edgecolor='k')
        plt.xlabel('Canonical Variates')
        plt.ylabel('Correlation Coefficient')
        plt.savefig(f'{self.output_path}/scree_plot.png')
        plt.show()

    def scatter_plot(self):
        plt.figure(figsize=(20, 10))
        for i in range(self.n_comp):
            plt.subplot(2, 5, i + 1)
            plt.scatter(self.X1_c[:, i], self.X2_c[:, i], alpha=0.7, label=f'Canonical Correlation {i+1}')
            slope, intercept = np.polyfit(self.X1_c[:, i], self.X2_c[:, i], 1)
            plt.plot(self.X1_c[:, i], slope * self.X1_c[:, i] + intercept, color='red')
            r, p_value = pearsonr(self.X1_c[:, i], self.X2_c[:, i])
            plt.xlabel(f'Canonical Variate {i+1} (Metagenomics)')
            plt.ylabel(f'Canonical Variate {i+1} (Metabolomics)')
            plt.title(f'CC {i+1}: r={r:.2f}, p={p_value:.3e}')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_path}/scatter_plots.png')
        plt.show()

    def plot_weights(self, df1, df2):
        # Limit to top 40 features if there are more than 40 columns
        if df1.shape[1] > 40:
            feature_importance_df1 = np.abs(self.cca.x_weights_).mean(axis=1)
            top_40_indices_df1 = np.argsort(feature_importance_df1)[-40:]
            df1 = df1.iloc[:, top_40_indices_df1]
            self.cca.x_weights_ = self.cca.x_weights_[top_40_indices_df1, :]

        if df2.shape[1] > 40:
            feature_importance_df2 = np.abs(self.cca.y_weights_).mean(axis=1)
            top_40_indices_df2 = np.argsort(feature_importance_df2)[-40:]
            df2 = df2.iloc[:, top_40_indices_df2]
            self.cca.y_weights_ = self.cca.y_weights_[top_40_indices_df2, :]

        feature_names_df1 = df1.columns
        feature_names_df2 = df2.columns

        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        for i in range(3):
            axs[i, 0].bar(feature_names_df1, self.cca.x_weights_[:, i])
            axs[i, 0].set_title(f'Canonical Weights of X for CC{i+1}')
            axs[i, 0].tick_params(axis='x', rotation=90)
            axs[i, 1].bar(feature_names_df2, self.cca.y_weights_[:, i])
            axs[i, 1].set_title(f'Canonical Weights of Y for CC{i+1}')
            axs[i, 1].tick_params(axis='x', rotation=90)

        plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
        plt.subplots_adjust(left=0.05, right=0.95, top=.95, bottom=.05, wspace=.35, hspace=1.0)
        plt.savefig(f'{self.output_path}/canonical_weights.png')
        plt.show()

    def heatmap(self, df1, df2, birdman_metag):
        coef_df = pd.DataFrame(np.round(self.cca.coef_, 2), columns=[df1.columns])
        coef_df.index = df2.columns
        coef_df = coef_df.T
        coef_df = coef_df.rename_axis('Feature').reset_index()
        coef_df = pd.merge(coef_df, birdman_metag[['Feature', 'Group']], on='Feature', how='left').set_index('Feature')
        group_labels_df = pd.DataFrame(coef_df['Group'])

        num_species = len(coef_df.index)
        num_biomarkers = len(coef_df.columns)
        bonferroni_alpha = 0.05 / (num_species * num_biomarkers)

        fig, ax = plt.subplots(figsize=(38, 30))
        group_numeric = group_labels_df['Group'].map({'Top': 1, 'Bottom': 0})
        coef_df['BIRDMAn Group'] = group_numeric.reindex(coef_df.index)
        heatmap = sns.heatmap(coef_df.iloc[:, :-2], ax=ax, annot=False, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Coefficient', 'pad': 0.02}, center=0)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Coefficient', size=20)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cax.imshow(coef_df[['BIRDMAn Group']].values, aspect='auto', cmap=ListedColormap(['lightgreen', 'pink']))
        cax.set_yticks(np.arange(coef_df.shape[0]))
        cax.set_yticklabels(coef_df.index)
        cax.yaxis.set_ticks_position('left')
        cax.set_xticks([])
        cax.set_xticklabels([])
        cax.set_yticklabels([])

        legend_patches = [mpatches.Patch(color='pink', label='Top'), mpatches.Patch(color='lightgreen', label='Bottom')]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.33, 1), fontsize=24, title="BIRDMAn Group", title_fontsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.tick_params(axis='x', labelsize=24)
        ax.set_xlabel('Biomarker', size=24)
        ax.set_ylabel('Feature', size=24)

        plt.tight_layout()
        plt.savefig(f'{self.output_path}/heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        
    def metadata_correlations(self, df1, df2, md, metadata_columns):
        columns = [f"CV{i+1}" for i in range(self.n_comp)]
        cv_df1 = pd.DataFrame(self.X1_c, df1.index, columns=columns)
        cv_df2 = pd.DataFrame(self.X2_c, df2.index, columns=columns)

         for col in metadata_columns:
            if md[col].dtype == 'object':  # Only apply to non-numeric columns
                unique_values = sorted(md[col].dropna().unique(), reverse=True)
                print(f"{col} Groups: {unique_values}")
                mapping_dict = {value: index for index, value in enumerate(unique_values)}
                md[col] = md[col].map(mapping_dict)
                md[col] = pd.to_numeric(md[col], errors='coerce')
                
                def perform_logistic_regression(cv, metadata_columns):
                    results = []
                    for group in metadata_columns:
                        X = cv
                        X = sm.add_constant(X)
                        y = md[group]
                        
                        y = y.dropna()
                        common_idx = X.index.intersection(y.index)
                        X = X.loc[common_idx]
                        y = y.loc[common_idx]

                        model = sm.Logit(y, X).fit(disp=0)
                        coef = model.params['CV1']
                        p_value = model.pvalues['CV1']
                        results.append((group, coef, p_value))
                    return pd.DataFrame(results, columns=['Group', 'Coefficient', 'P-value'])

                results_df1 = perform_logistic_regression(cv_df1['CV1'], metadata_columns)
                results_df2 = perform_logistic_regression(cv_df2['CV1'], metadata_columns)
            # else:
                
            
        results_df1['Source'] = 'DF1'
        results_df2['Source'] = 'DF2'
        results = pd.concat([results_df1, results_df2])
        results['Significant'] = results['P-value'] < 0.05

        plt.figure(figsize=(12, 8))
        sns.barplot(data=results, x='Group', y='Coefficient', hue='Source', palette=['blue', 'green'])

        for i, row in results.iterrows():
            if row['Significant']:
                plt.text(i, row['Coefficient'], '*', ha='center', va='bottom', color='black', fontsize=12)

        plt.title('Coefficients of CV1 related to Phenotypes')
        plt.xlabel('Group')
        plt.ylabel('Coefficient')
        plt.legend(title='Data Source')
        plt.savefig(f'{self.output_path}/phenotypes.png')
        plt.show()

# In[ ]:




