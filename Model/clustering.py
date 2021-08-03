import pickle
import numpy as np
from matplotlib.pyplot import figure
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from utils import sort_and_select_10_neigh_dist

with open("outputs/target_texts.pickle", 'rb') as handle:
    target_texts = pickle.load(handle)

tqdm.pandas()
target_similarity_matrix = pd.read_csv("outputs/target_similarity_matrix.txt", header=None)

neighbor_10_dist = target_similarity_matrix.apply(sort_and_select_10_neigh_dist, axis = 1)

for x in list(target_similarity_matrix.nsmallest(10, 1).index.astype(int)):
    print(target_texts[x], " - Similarity Score :", target_similarity_matrix[x][1])

eps_10 = neighbor_10_dist.quantile(0.99)

# eps_nums = list()
# eps_num_clusters = list()
# eps_num_noise = list()
# trial_eps = np.linspace(0, 0.2, num=100)

# for eps_num in trial_eps:
    
#     if eps_num > 0:   
            
#         print("Taking EPS as", eps_num)
#         # Compute DBSCAN
#         db = DBSCAN(eps=eps_num, min_samples=2, metric="precomputed", n_jobs=4).fit(target_similarity_matrix)
#         labels = db.labels_

#         # Number of clusters in labels, ignoring noise if present.
#         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#         n_noise_ = list(labels).count(-1)

#         eps_nums.append(eps_num)
#         eps_num_clusters.append(n_clusters_)
#         eps_num_noise.append(n_noise_)

#         # print('Estimated number of clusters: %d' % n_clusters_)
#         # print('Estimated number of noise points: %d' % n_noise_)
#         # print("-----------")
        
#         if n_clusters_ == 1:
#             break

# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('eps')
# ax1.set_ylabel('number of clusters', color=color)
# ax1.plot(eps_nums, eps_num_clusters, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()

# ax2.plot(eps_nums, eps_num_noise)
# ax2.set_ylabel('noise')
# plt.show()

eps_num = 0.0025
print("Taking EPS as", eps_num)
# Compute DBSCAN
db = DBSCAN(eps=eps_num, min_samples=3, metric="precomputed", n_jobs=4).fit(target_similarity_matrix)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


for unique_label in set(labels):
    
    class_member_mask = (labels == unique_label)
    print("In cluster", unique_label, "found", Counter(class_member_mask)[True], "points")
    print("Samples")
    print(np.array(target_texts)[class_member_mask])
    print("-------------------------------------")

with open('outputs/target_dbscan.pickle', 'wb') as handle:
    pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)