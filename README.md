# Person Re-identification: Estimate the number of unique people

## Problem Statement

Given a sample of 25000 crops taken from a person re-id dataset. The task of the problem is to use deep learning techniques to cluster and identify the no. of unique people in this dataset. 

**Task 1:** Estimate the no. of unique people in the dataset

**Task 2** Visualise the clusters of people and the activation maps of the person re-id features

## Summary

This project explores the task of clustering images of persons into distinct individuals using a combination of ResNet(Transfer Learning) and Autoencoders as feature extractors, along with KMeans and DBSCAN as clustering algorithms. The aim is to leverage advanced techniques to identify and group similar images, ultimately determining the total number of individuals present in the dataset.

### Feature Extraction

Autoencoders are used as the primary feature extraction technique. The Autoencoder model is trained to reconstruct the input images, forcing it to learn a compact representation in the process. The bottleneck layer of the Autoencoder captures essential features specific to each person, which are then utilized for clustering. Since the bottleneck layer has lower dimensionality, it reduces the computational cost sizably.

Additionally, ResNet, a deep convolutional neural network architecture, is also employed as a feature extractor to capture high-level representations from the person images. By utilizing a pre-trained ResNet model, the project benefits from its ability to learn complex visual patterns and extract meaningful features. These extracted features form the foundation for subsequent clustering analysis.

### Clustering with extracted features:
The project applies clustering algorithms to group the extracted features based on their similarities. KMeans, a centroid-based clustering algorithm, partitions the feature space into distinct clusters. Elbow method analysis is used to identify the number of unique clusters.

Furthermore, DBSCAN, a density-based clustering algorithm, is explored. DBSCAN identifies dense regions in the feature space and groups data points that are closely packed together. Unlike KMeans, DBSCAN does not require specifying the number of clusters in advance, making it suitable for scenarios where the total number of individuals is unknown. DBSCAN also is not constrained with identifying circular clusters the way KMeans is, this gives it an added advantage to be able to cluster non-circular shapes also.

### Results and Evaluation:
AutoEncoders were utilized in the final clustering. Since the images did not have labels, no additional layer could be added to the Pre-Trained ResNet to reduce the dimensions of the extracted features, AutoEncoders were used to reduce the dimensionality of the images to a 32 size vector. 

The final Feature Extractor employs 6 layers, 3 Conv layers and 3 Dense layers in the encoder, and 3 dense layers followed by 3 convolution layers for the reconstruction in the decoder. LeakyReLU activation function was used in all layers except the last softmax layer. The Model used Mean Absolute Error as loss function and Learning Rate of 0.001. Early stopping with patience of 20 epocs was employed to reduce training time and chances of overfitting. Due to resource constraints, the training was done in parts of 7000 images at once, for 100 epocs and with batch size 64.

The latent layer of the autoencoder was of size 32 pixels, and the resulting features extracted are of the same dimensions.

KMeans was employed to find the number of unique clusters. Elbow method was applied to estimate number of clusters where inter-cluster distance is maximum while intra-cluter distance is minimum. But since KMeans can only estimate spherical cloud distributions and elbow was forming at quite a low number, DBSCAN was applied to find the final number of features.

Taking two similar images, the Euclidean distance between their features was calculated, based on which eps value for the model was set. The resultant produces 1302 clusters of unique persons in the dataset.

## Directory Structure
The project repository consists of the following main folders:

- Research: Contains notebooks for different versions of ResNet and AutoEncoders used for feature extraction. It also includes generated feature files and elbow graphs for KMeans clustering.

- Inference: Contains two subfolders: "src" and "results".

  src: Contains notebooks and visualizations related to the inference process. 
  - AutoEncoderV5 notebook for feature extraction.
  - DBSCAN clustering notebook
  - Netron visualizations of encoder and decoder models
  - Activation maps of the encoder.

  results: Contains output and analysis of the clustering process. 
  - Final cluster labels file mapping filenames to cluster labels
  - Sample clusters for reference
  - "Visualize_clusters.ipynb" notebook for generating image plots based on cluster ID
  - "Clusters" subfolder organizing images by clusters.

- Models: Holds saved models for all five autoencoder versions used in the project. It includes the respective encoder and decoder versions.

## Further improvements
- Intensive Hyperparameter Tuning: Due to resource constraints, the hyperparameter tuning process might have been limited. To optimize the models and achieve better results, a more extensive hyperparameter tuning can be performed. This involves systematically exploring various combinations of hyperparameters to identify the optimal settings for the models.

- Improving Autoencoder and Feature Latent Space: The quality of the compressed features extracted by the autoencoder can significantly impact the clustering results. Fine-tuning the autoencoder architecture and training process can lead to better feature representations. Techniques such as adding regularization, adjusting layer sizes, or incorporating advanced architectural modifications can be explored to improve the latent space and enhance the quality of the extracted features.

- Utilizing Siamese Neural Network: To further enhance the clustering performance, a Siamese Neural Network (SNN) can be utilized. The clusters obtained in the current implementation can serve as a base for training an SNN using triplets of similar and dissimilar images. By learning from the relationships between images, the SNN can generate better separation and discrimination among individuals, leading to improved clustering results.

- Reclassification of Outliers or Wrongly Classified Images: In the clustering process, there might be instances where outliers or images wrongly classified into clusters are present. To refine the clustering results, a reclassification step can be performed. This involves manually reviewing and reassigning outliers or misclassified images based on domain knowledge or additional labelling information. By iteratively refining the clustering results, the overall accuracy and reliability of the final clusters can be improved.
