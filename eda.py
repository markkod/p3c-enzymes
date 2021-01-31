from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.sparse import load_npz
import auxiliarymethods.auxiliary_methods as aux
from auxiliarymethods import datasets as dp
from auxiliarymethods.reader import tud_to_networkx
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from nrkmeans import NrKmeans
import networkx as nx

def load_csv(path):
    return np.loadtxt(path, delimiter=";")

def load_sparse(path):
    return load_npz(path)

def visualize(G, color=None, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, 
                     pos=nx.spring_layout(G, seed=42),
                     with_labels=True,
                     node_color=color,
                     cmap="Set2")
    plt.show();

def plot_wl_nmi_comparison():
  algorithms = ["KPCA", "TSVD", "SPEC"]
  for version in ["node_labels", "without_labels"]:
      print(f"#################{version}#################")
      all_nmi = {}

      for algorithm in algorithms:
          path_i = os.path.join('kernels', version, f'{algorithm}.csv')
          nmi_df = pd.read_csv(path_i, index_col=0)
          all_nmi[algorithm] = nmi_df['ENZYMES']

          max_nmi = nmi_df['ENZYMES'].max()
          max_nmi_id = nmi_df['ENZYMES'].idxmax()

      all_nmi_df = pd.DataFrame.from_dict(all_nmi)
      # print(all_nmi_df)

      fig, ax = plt.subplots(figsize=(10,3))
      ax.set_ylabel("NMI")
      ax.set_xlabel("WL-Iterations")
      ax.set_xticks([0,1,2,3,4,5,6])
      ax.set_xticklabels([1,2,3,4,5, 'graphlet', 'shortestpath'])
      ax.set_title(version)
      all_nmi_df.plot(marker="o", ax=ax)
      plt.show()


def plot_representations_nmi_comparison():
  rc = pd.read_csv('representation_comparison.csv')
  rc_node_labels = rc[rc.Labels == True]
  rc_without_labels = rc[rc.Labels == False]

  fig, axs = plt.subplots(1,3, figsize=(15,4))

  for (i, algorithm) in enumerate(rc.Algorithm.unique()):
    df = rc[rc.Algorithm==algorithm]
    axs[i].set_ylabel("NMI")
    axs[i].set_xlabel("Representation")
    axs[i].set_title(algorithm)
    axs[i].set_ylim([0,0.1])

    sns.barplot(x=df['Representation'], y=df["NMI"], data=df, hue='Labels', ax=axs[i])

  plt.show()


def plot_kpca_nmi_and_clustering(classes):
  ds_name = 'ENZYMES'
  base_path = os.path.join("kernels","node_labels")

  fig, axs = plt.subplots(3,3, figsize=(15,15))
  representations = ["wl3", "graphlet", "shortestpath"]

  for (i, representation) in enumerate(representations): 
    gram = load_csv(os.path.join(base_path,f"{ds_name}_gram_matrix_{representation}.csv"))
    gram = aux.normalize_gram_matrix(gram)

    kpca = KernelPCA(n_components=100, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(gram)
    # fig, ax = plt.subplots(figsize=(5,5))
    axs[0][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=classes, s=1)
    axs[0][i].set_title(f'{representation} KPCA ground truth')
    
    kmeans = KMeans(n_clusters=6).fit(reduced_kpca)
    axs[1][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=kmeans.labels_, s=1)
    axs[1][i].set_title(f'{representation} KPCA KMeans')
    print(f"NMI KMeans {representation}: {nmi(classes, kmeans.labels_)}")

    db = DBSCAN().fit(reduced_kpca)
    axs[2][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=db.labels_, s=1)
    axs[2][i].set_title(f'{representation} DBSCAN KMeans')
    print(f"NMI DBSCAN {representation}: {nmi(classes, db.labels_)}\n")

  plt.show()


def plot_tsvd_nmi_and_clustering(classes):
  ds_name = 'ENZYMES'
  base_path = os.path.join("kernels","node_labels")

  fig, axs = plt.subplots(3,3, figsize=(15,15))
  representations = ["wl3", "graphlet", "shortestpath"]

  for (i, representation) in enumerate(representations): 
      vec = load_sparse(os.path.join(base_path,f"{ds_name}_vectors_{representation}.npz"))
      tsvd = TruncatedSVD(n_components=100)
      reduced_tsvd = tsvd.fit_transform(vec)

      # fig, ax = plt.subplots(figsize=(5,5))
      axs[0][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=classes, s=1)
      axs[0][i].set_title("Representation: " + representation)
      axs[0][i].set_title(f'{representation} TSVD')

      kmeans = KMeans(n_clusters=6).fit(reduced_tsvd)
      axs[1][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=kmeans.labels_, s=1)
      axs[1][i].set_title(f'{representation} TSVD KMeans')
      print(f"NMI KMeans {representation}: {nmi(classes, kmeans.labels_)}")

      db = DBSCAN().fit(reduced_tsvd)
      axs[2][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=db.labels_, s=1)
      axs[2][i].set_title(f'{representation} DBSCAN KMeans')
      print(f"NMI DBSCAN {representation}: {nmi(classes, db.labels_)}\n")
  plt.show();

def compute_different_nmis():
  reduced_kpca_dict = {}
  reduced_tsvd_dict = {}

  base_path = os.path.join("kernels","node_labels")
  dataset = "ENZYMES"
  nmis_tsvd = {}
  nmis_spec = {}
  nmis_kpca = {}

  classes = dp.get_dataset(dataset)
  representations = ["wl1", "wl2", "wl3", "wl4", "wl5", "graphlet", "shortestpath"]

  for representation in representations:

      #Gram Matrix for the Weisfeiler-Lehman subtree kernel
      gram = load_csv(os.path.join(base_path,f"{dataset}_gram_matrix_{representation}.csv"))
      gram = aux.normalize_gram_matrix(gram)

      #Sparse Vectors for the Weisfeiler-Lehmann subtree kernel
      vec = load_sparse(os.path.join(base_path,f"{dataset}_vectors_{representation}.npz"))

      tsvd = TruncatedSVD(n_components=100)
      reduced_tsvd = tsvd.fit_transform(vec)

      kpca = KernelPCA(n_components=100, kernel="precomputed")
      reduced_kpca = kpca.fit_transform(gram)

      reduced_kpca_dict[f'{representation}'] = reduced_kpca
      reduced_tsvd_dict[f'{representation}'] = reduced_tsvd

      k = len(set(classes.tolist()))
      d = {0:"TSVD",1:"KPCA"}
      n_d = {0:nmis_tsvd, 1:nmis_kpca}
      for i,rep_i in enumerate([reduced_tsvd, reduced_kpca]):
          

          # fig, ax = plt.subplots(figsize=(5,5))
          # ax.scatter(rep_i[:,0], rep_i[:,1], c=classes, s=1)
          # plt.show();

          # Apply Subkmeans
          nrkm = NrKmeans(n_clusters=[k,1])#, allow_larger_noise_space=False)
          nrkm.fit(rep_i, best_of_n_rounds=10, verbose=False)
          subkm_nmi = nmi(nrkm.labels[0],classes)
          n_d[i][representation] = subkm_nmi
          # Plot rotated space
          V = nrkm.V
          rotated = np.dot(rep_i,V)
          reduced_df = pd.DataFrame(rotated[:,0:2])
          reduced_df["labels"] = classes#nrkm.labels[0]
          # sns.pairplot(reduced_df, hue="labels", diag_kind="hist")
          # plt.show();

      # Apply Spectral Clustering
      spec = SpectralClustering(n_clusters=k, affinity="precomputed")
      spec.fit(gram)
      spec_nmi = nmi(spec.labels_,classes)
      nmis_spec[representation] = spec_nmi

  res = {"KPCA":nmis_kpca,"TSVD": nmis_tsvd, "SPEC": nmis_spec}
  for key, value in res.items():
      print(key)
      for (representation, nmi_) in value.items():
          print(representation, ":", nmi_)