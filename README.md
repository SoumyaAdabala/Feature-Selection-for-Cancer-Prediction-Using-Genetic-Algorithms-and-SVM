# Feature-Selection-for-Cancer-Prediction
Applied GA-based feature selection on cancer datasets, reducing features from 30 to 14 and boosting SVM accuracy from 86% to 92%. • . • Enhanced model generalization by leveraging GA-optimized feature subsets, reducing dimensionality while improving scalability and efficiency


Here is a README file for GitHub, based on the provided sources:

---

# **Genetic Algorithm for Feature Selection in Cancer Prediction**

## **Project Overview**

This project demonstrates the application of **Genetic Algorithms (GA) for feature selection in training machine learning (ML) models for cancer prediction**. The primary goal is to **optimise the selection of the most relevant features** from a given dataset to improve model performance, reduce dimensionality, and increase computational efficiency. This addresses the "Curse of dimensionality," where unnecessary data leads to excessive complications and can create uncertainty about cancer prognosis.

Genetic testing analyses DNA to find mutations linked to cancer risk, generating bioinformatics data with diverse features like gene expressions and protein sequences. Handling such complex data efficiently is crucial. **Genetic Algorithms (GA) are employed in feature selection to optimise the selection of relevant features by evolving subsets that maximise model performance**. GAs help in reducing dimensionality by identifying the most significant features, thereby improving computational efficiency and model accuracy.

## **Why Genetic Algorithms?**

Genetic Algorithms are chosen for feature selection because they are:
*   **Flexible**
*   **Adaptable**
*   **Robust**
*   **Efficiently handle complex and nonlinear feature interactions**

## **Key Components**

*   **Feature Selection**: The core solution to the "Curse of dimensionality," involving selecting the right subset of data to improve computational efficiency and model accuracy.
*   **Genetic Algorithms (GA)**: Used to evolve subsets of features, aiming to maximise model performance.
*   **Support Vector Machine (SVM)**: The machine learning algorithm used to classify cancer data and evaluate the performance of selected features. It is used for model training and evaluation within the fitness function.
*   **Fitness Function**:
    *   **Objective**: To evaluate the performance of feature subsets based on classification accuracy.
    *   **Inputs**: Feature Matrix X (n × m samples x features), Target Labels y (n × 1), and Selected Features s (m × 1 binary vector).
    *   **Evaluation**: Uses **k-fold cross-validation (specifically 5-fold cross-validation)** to evaluate SVM model performance and compute accuracy for each fold. The fitness score is the average accuracy across all folds.

## **Dataset**

*   **`cancer_data.mat`**:
    *   This dataset contains cancer-related features and their corresponding labels.
    *   It comprises **30 features and an associated class label for each sample**.

## **Program Code Files**

*   **`GA_FS_CD.m`**:
    *   A MATLAB script that implements a Genetic Algorithm for feature selection.
    *   It loads the `cancer_data.mat` dataset and uses GA to select optimal features, enhancing the accuracy of the SVM model.

## **Results and Visualisation**

The script generates plots to visualise the project's outcomes, including:
*   **Fitness score over generations**
*   **Feature selection across generations**
*   **Frequency of feature selection**

## **Applications**

The principles and methodology demonstrated in this project have broad applications, including:
*   **Healthcare Diagnosis**
*   **Customer Segmentation**
*   **Predictive Maintenance**

## **References**

*   Liu, B. G., Xu, L. J., Wang, Y. H., & Tang, J. H. (2012). Genetic Algorithm-Based Feature Selection for Classification: A Comparative Study. Presented at the IEEE International Conference on Systems, Man, and Cybernetics (SMC). Available at IEEE Xplore.
*   García, J. S., Gómez, J. A., & Carrillo, A. J. L. (2013). Feature Selection Using Genetic Algorithms: A Review. IEEE Transactions on Evolutionary Computation. Available at IEEE Xplore.
*   Pal, S. K., Sinha, S. K., & Chaudhuri, B. B. (2011). Feature Selection for Classification Using Genetic Algorithms. Pattern Recognition. Available at ScienceDirect.

---
