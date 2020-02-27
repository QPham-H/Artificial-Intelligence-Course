from __future__ import division
import numpy as np

# How it works (I think)
# 1) After selecting k random points to initialize
# 2) Run through each point and find the Euclidean distance
# 3) Assign to closest k cluster and add to its running sum (each dimension separately)
# 4) Find the new mean (for each dimension)
# Repeat 2-4 for given number of iterations

def k_means (k, iters, d_array):
    (rows,cols) = d_array.shape

    # Initialize with random k points
    rand_k = np.random.choice(range(rows), k, replace=False)
    mean_array = np.empty([k,cols])
    for idx, pt in enumerate(rand_k):
        mean_array[idx,:] = d_array[pt,:]
        
    for i in range(iters-1):
        SS_total = np.zeros([k,1])
        new_mean_sum = np.zeros([k,cols])
        k_members = np.zeros([k,1])
        dist_array = np.empty([k,1])
        k_mem_list = np.empty(shape=(rows,1))
        for idx, row in enumerate(d_array):
            for k_idx,k_row in enumerate(mean_array):
                dist_array[k_idx] = np.sum(np.square(row-k_row))
            closest_k = np.argmin(dist_array, axis=0)
            SS_total[closest_k] += dist_array[closest_k]
            new_mean_sum[closest_k] += row
            k_members[closest_k] += 1
            k_mem_list[idx] = closest_k
        mean_array = new_mean_sum / k_members

    return (np.sum(SS_total), k_mem_list, mean_array)

# Retrieve data from known source of 150
setosa = 0
versicolor = 0
virginica = 0
cluster_data = np.empty([150,4])
cluster_name = []
with open('iris.data') as data:
    for idx, line in enumerate(data):
        raw_data = line.strip().split(',')
        cluster_data[idx] = [raw_data[0], raw_data[1],
                             raw_data[2], raw_data[3]]
        cluster_name.append(raw_data[4])
        if(raw_data[4] == 'Iris-setosa'):
            setosa += 1
        elif(raw_data[4] == 'Iris-versicolor'):
            versicolor += 1
        else:
            virginica += 1

# Change the parameters here for each trial
# ||
# \/
best_cluster = np.empty(shape=(150,1))
best_SS = 999
for i in range(4):
    k = 3
    iters = 20
    best_centroids = np.empty(shape=(k,4))
    (SS_total, mem_list, centroids) = k_means(k, iters, cluster_data)
    if(SS_total <= best_SS):
        best_SS = SS_total
        best_cluster = mem_list
	best_centroids = centroids
    print ('This is cluster calculation #',i,
           ' for k=',k ,' and iterations=',iters)
    print ('SS_total=',SS_total)
    
print ("Best SS_total=", best_SS, " with centroids at:")
print (best_centroids)
print ("And with cluster members:")
print (best_cluster)

# For Iris_setosa
(k_val, freq_array) = np.unique(best_cluster[0:(setosa-1)], return_counts=True)
idx = np.argmax(freq_array)
correct_seto = freq_array[idx]
seto_rep = k_val[idx]
predicted_setosa = np.count_nonzero(best_cluster == seto_rep)
recall = correct_seto / setosa
precision = correct_seto / predicted_setosa
F1_seto = 2*precision*recall/(precision + recall)
print ("Correct seto count =", correct_seto)
print ("Setosa example =",setosa,
       "while predicited_setosa was =",predicted_setosa)
print ("Recall =",recall, "and Precision =",precision)
print ("F1 =",F1_seto)

# For Iris_versicolor
(k_val, freq_array) = np.unique(best_cluster[(setosa-1):(setosa+versicolor-1)], return_counts=True)
idx = np.argmax(freq_array)
correct_vers = freq_array[idx]
vers_rep = k_val[idx]
predicted_versicolor = np.count_nonzero(best_cluster == vers_rep)
recall = correct_vers / versicolor
precision = correct_vers / predicted_versicolor
F1_vers = 2*precision*recall/(precision + recall)
print ("Correct vers count =", correct_vers)
print ("Versiolor example =",versicolor,
       "while predicted_versicolor was =",predicted_versicolor)
print ("Recall =",recall, "and Precision =",precision)
print ("F1 =",F1_vers)

# For Iris_virginica
(k_val, freq_array) = np.unique(best_cluster[(setosa+versicolor-1):], return_counts=True)
idx = np.argmax(freq_array)
correct_virg = freq_array[idx]
virg_rep = k_val[idx]
predicted_virginica = np.count_nonzero(best_cluster == virg_rep)
recall = correct_virg / virginica
precision = correct_virg / predicted_virginica
F1_virg = 2*precision*recall/(precision + recall)
print ("Correct virg count =", correct_virg)
print ("Virginica example =",virginica,
       "while predicted_virginica was =",predicted_virginica)
print ("Recall =",recall, "and Precision =",precision)
print ("F1 =",F1_virg)

F1_avg = (F1_seto + F1_vers + F1_virg) / 3.0
print ("Average F1 of clustering =",F1_avg)

# To plot the data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()

# To getter a better understanding of interaction of the dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(cluster_data[:, 0], cluster_data[:, 1],
           cluster_data[:, 2], c=(best_cluster/100),
           cmap=plt.cm.Paired)
ax.scatter(best_centroids[:, 0], best_centroids[:, 1],
           best_centroids[:, 2], marker='*',
           cmap=plt.cm.Paired, s=200, alpha=1)
ax.set_title("Sepal Length vs Sepal Width vs Petal Length")
ax.set_xlabel("sepal length")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("sepal width")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("petal length")
ax.w_zaxis.set_ticklabels([])

plt.show()

fig = plt.figure(2, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(cluster_data[:, 0], cluster_data[:, 1],
           cluster_data[:, 3], c=(best_cluster/12),
           cmap=plt.cm.Paired)
ax.scatter(best_centroids[:, 0], best_centroids[:, 1],
           best_centroids[:, 3], marker='*',
           cmap=plt.cm.Paired, s=200, alpha=1)
ax.set_title("Sepal Length vs Sepal Width vs Petal Width")
ax.set_xlabel("sepal length")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("sepal width")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("petal width")
ax.w_zaxis.set_ticklabels([])

plt.show()

fig = plt.figure(3, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(cluster_data[:, 0], cluster_data[:, 2],
           cluster_data[:, 3], c=(best_cluster/12),
           cmap=plt.cm.Paired)
ax.scatter(best_centroids[:, 0], best_centroids[:, 2],
           best_centroids[:, 3], marker='*',
           cmap=plt.cm.Paired, s=200, alpha=1)
ax.set_title("Sepal Length vs Petal Length vs Petal Width")
ax.set_xlabel("sepal length")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("petal length")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("petal width")
ax.w_zaxis.set_ticklabels([])

plt.show()

fig = plt.figure(4, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(cluster_data[:, 1], cluster_data[:, 2],
           cluster_data[:, 3], c=(best_cluster/12),
           cmap=plt.cm.Paired)
ax.scatter(best_centroids[:, 1], best_centroids[:, 2],
           best_centroids[:, 3], marker='*',
           cmap=plt.cm.Paired, s=200, alpha=1)
ax.set_title("Sepal Width vs Petal Length vs Petal Width")
ax.set_xlabel("sepal length")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("petal length")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("petal width")
ax.w_zaxis.set_ticklabels([])

plt.show()
