# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans , DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer


train_df = pd.read_csv(r'C:\Users\aliel\Downloads\flight_train.csv')
test_df = pd.read_csv(r'C:\Users\aliel\Downloads\flight_test.csv')
merged_df = pd.concat([train_df, test_df], ignore_index=True)

#EDA

merged_df.info()
merged_df.isna().sum()
merged_df.nunique()

sns.histplot(merged_df['AGE'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

sns.countplot(x='GENDER', data=merged_df)
plt.title('Gender Distribution')
plt.show()

sns.boxplot(x=merged_df['FLIGHT_COUNT'])
plt.title('Flight Distribution')
plt.show()

merged_df['FFP_DATE'] = pd.to_datetime(merged_df['FFP_DATE'])
merged_df.groupby(merged_df['FFP_DATE'].dt.to_period('M')).size().plot()
plt.title('Customer Activity Over Time')
plt.ylabel('Number of Customers')
plt.xlabel('Month')
plt.show()

sns.countplot(x='FFP_TIER', data=merged_df)
plt.title('Loyalty Tier Distribution')
plt.show()

corr = merged_df[['AGE', 'FLIGHT_COUNT', 'BP_SUM', 'SUM_YR_1', 'SUM_YR_2', 'SEG_KM_SUM', 'AVG_INTERVAL', 'MAX_INTERVAL', 'EXCHANGE_COUNT', 'avg_discount']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Preprocessing
merged_df['GENDER'].fillna(merged_df['GENDER'].mode()[0], inplace=True)
merged_df['WORK_CITY'].fillna('Unkown', inplace=True)
merged_df['WORK_PROVINCE'].fillna('Unkown', inplace=True)
merged_df['WORK_COUNTRY'].fillna('Unkown', inplace=True)
merged_df['AGE'].fillna(merged_df['AGE'].median(), inplace=True)
merged_df['SUM_YR_1'].fillna(merged_df['SUM_YR_1'].median(), inplace=True)
merged_df['SUM_YR_2'].fillna(merged_df['SUM_YR_2'].median(), inplace=True)

merged_df.info()

merged_df = pd.get_dummies(merged_df, columns=['GENDER', 'WORK_CITY', 'WORK_PROVINCE',
                                               'WORK_COUNTRY'], drop_first=True)
#Scaling
scaler = StandardScaler()

numerical_features = [ 'FLIGHT_COUNT', 'BP_SUM', 'SUM_YR_1', 'SUM_YR_2', 'SEG_KM_SUM']

merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])

#MODEL 1 KMeans
inertia = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(merged_df[numerical_features])
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11), inertia, marker='o')
plt.title('Elbow Method')
plt.ylabel('Inertia')
plt.xlabel('Number Of Clusters')
plt.show()


sil_scores = []
for i in range(2,11):
    kmeans1 =  KMeans(n_clusters=i, init='k-means++', n_init=20,
                      random_state=42).fit(merged_df[numerical_features])
    labels = kmeans1.labels_
    sil_scores.append(silhouette_score(merged_df[numerical_features], labels))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.xlabel('K')
plt.ylabel('Silhouette score')
plt.title('Optimal K')
plt.show()

kmeans = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
merged_df['KMeans_cluster'] = kmeans.fit_predict(merged_df[numerical_features])

sns.scatterplot(data=merged_df, x='AGE', y='FLIGHT_COUNT', hue='KMeans_cluster', palette='viridis')
plt.title('KMeans Clustering (AGE Vs FLIGHT COUNT)')
plt.ylabel('FLIGHT COUNT')
plt.xlabel('AGE')
plt.show()


#MODEL 2 DBSCAN
min_samples = 3

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(merged_df[numerical_features])
distances, indices = neighbors_fit.kneighbors(merged_df[numerical_features])

distances = np.sort(distances[:, min_samples-1, ], axis=0)

plt.plot(distances)
plt.xlabel('Points')
plt.ylabel(f'{min_samples}_th Nearst Neighbor Graph')
plt.title('K-distance graph ')
plt.show()

dbscan = DBSCAN(eps=0.5 , min_samples=3 )
merged_df['DBSCAN_cluster'] = dbscan.fit_predict(merged_df[numerical_features])


sns.scatterplot(data=merged_df, x='AGE', y='FLIGHT_COUNT', hue='DBSCAN_cluster',
                palette='viridis', marker='o')
plt.title('DBSCAN Clustering (AGE Vs FLIGHT COUNT)')
plt.ylabel('FLIGHT COUNT')
plt.xlabel('AGE')
plt.show()

#MODEL 3 PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(merged_df[numerical_features])

kmeans_pca = KMeans(n_clusters=2, random_state=42)
labels_pca = kmeans_pca.fit_predict(pca_components)

dbscan_pca = DBSCAN(eps=0.5, min_samples=3)
merged_df['DBSCAN_pca_cluster'] = dbscan_pca.fit_predict(pca_components)


merged_df['PCA1'] = pca_components[:, 0]
merged_df['PCA2'] = pca_components[:, 1]

sns.scatterplot(data=merged_df, x='PCA1', y='PCA2', hue='KMeans_cluster',
                palette='viridis')
plt.title('PCA with DBSCAN Clusters')
plt.ylabel('PCA2')
plt.xlabel('PCA1')
plt.show()

#Model Evaluation
silhouette_dbscan = silhouette_score(merged_df[numerical_features], merged_df['DBSCAN_cluster'])
silhouette_kmeans = silhouette_score(merged_df[numerical_features], merged_df['KMeans_cluster'])
silhouette_pca = silhouette_score(pca_components, merged_df['KMeans_cluster'])
dbscan_pca_score = silhouette_score(pca_components, merged_df["DBSCAN_pca_cluster"])


print(f'DBSCAN silhouette score : {silhouette_dbscan}')
print(f'KMeans silhouette score : {silhouette_kmeans}')
print(f'PCA silhouette score : {silhouette_pca}')
print("DBSCAN with PCA silhouette score:", dbscan_pca_score)

# submit

test_df['GENDER'].fillna(test_df['GENDER'].mode()[0], inplace=True)
test_df['WORK_CITY'].fillna('Unkown', inplace=True)
test_df['WORK_PROVINCE'].fillna('Unkown', inplace=True)
test_df['WORK_COUNTRY'].fillna('Unkown', inplace=True)
test_df['AGE'].fillna(test_df['AGE'].median(), inplace=True)
test_df['SUM_YR_1'].fillna(test_df['SUM_YR_1'].median(), inplace=True)
test_df['SUM_YR_2'].fillna(test_df['SUM_YR_2'].median(), inplace=True)

test_df[numerical_features] = scaler.fit_transform(test_df[numerical_features])




test_pca = pca.transform(test_df[numerical_features])
submission = test_df[['MEMBER_NO']]
submission['Cluster'] = dbscan_pca.fit_predict(test_pca)

submission.to_csv('D:\Customer Segmentation.csv', index=False)

