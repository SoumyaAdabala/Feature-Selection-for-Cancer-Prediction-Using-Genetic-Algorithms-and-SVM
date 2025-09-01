# Feature-Selection-for-Cancer-Prediction
Applied GA-based feature selection on cancer datasets, reducing features from 30 to 14 and boosting SVM accuracy from 86% to 92%. • . • Enhanced model generalization by leveraging GA-optimized feature subsets, reducing dimensionality while improving scalability and efficiency


Here is a README file for GitHub, based on the provided sources:

---
Genetic Algorithm for Feature Selection in Cancer Prediction
This project demonstrates the application of a Genetic Algorithm (GA) to perform feature selection for a cancer prediction machine learning model. The goal is to identify the most relevant features in a dataset, thereby reducing dimensionality, improving model accuracy, and increasing computational efficiency. A Support Vector Machine (SVM) classifier is used to evaluate the fitness of each feature subset selected by the GA.

The Problem: Curse of Dimensionality
In machine learning, especially with bioinformatics data, datasets can contain a vast number of features (e.g., gene expressions, protein sequences). This "curse of dimensionality" can lead to several issues:

Overfitting: The model learns the training data too well, including noise, and performs poorly on new, unseen data.

Increased Computational Cost: More features mean more processing power and time are required to train the model.

Uncertainty: Unnecessary features can introduce noise and make it harder to interpret the model's predictions.

The Solution: Genetic Algorithms
Genetic Algorithms (GAs) are a powerful optimization technique inspired by the process of natural selection. They are well-suited for feature selection because they can efficiently search large and complex spaces to find an optimal subset of features. By evolving a population of feature subsets over generations, the GA identifies the features that contribute most to the model's predictive power.

This approach helps to:

Select the most significant features.

Reduce the complexity of the model.

Improve the accuracy and efficiency of the final classifier.

Files in This Repository
GA_FS_CD.m: The main MATLAB script that implements the Genetic Algorithm for feature selection.

cancer_data.mat: The dataset file containing a matrix X with 30 features for multiple samples and a vector y with the corresponding class labels.

How to Run the Code
Prerequisites: Ensure you have MATLAB installed on your system.

Clone Repository: Download or clone this repository to your local machine.

Update File Path: Open the GA_FS_CD.m script in the MATLAB editor. You must update the file path on line 2 to point to the location of cancer_data.mat on your computer.

% CHANGE THIS PATH
load("C:\path\to\your\folder\cancer_data.mat", 'X', 'y');


<img width="657" height="281" alt="image" src="https://github.com/user-attachments/assets/9dc8d5ed-c7fe-4ec5-8803-d9e2ddc54ebe" />

Execute: Run the GA_FS_CD.m script. The script will output the progress for each generation to the command window and display plots of the results upon completion.

Methodology
The algorithm follows the steps outlined in the flowchart below. It starts by loading the data and initializing a population of "chromosomes," where each chromosome represents a subset of features. In a loop, it evaluates the fitness of each subset using an SVM classifier with 5-fold cross-validation, then performs selection, crossover, and mutation to create a new generation of feature subsets. This process repeats for 50 generations.

Key GA Parameters:

Population Size: 50

Number of Generations: 50

Mutation Rate: 0.05

Number of Features to Select: 15 (half of the total features)

Results
After running for 50 generations, the algorithm identifies the best subset of features based on the highest achieved fitness score (classification accuracy).

<img width="1471" height="916" alt="Results" src="https://github.com/user-attachments/assets/707c2206-d660-47cd-a542-e9011ac48a9f" />



Command Window Output
The command window shows the best fitness score at each generation and lists the final set of selected features.

Final Selected Features
The following features were selected by the algorithm as the most optimal set:

3, 7, 8, 9, 10, 11, 16, 17, 18, 19, 21, 24, 26, 28, 29

Performance Plots
The following plots visualize the performance and behavior of the Genetic Algorithm during the selection process.

Fitness Score Over Generations: This plot shows the best fitness score (accuracy) in each generation. The fluctuations indicate the exploratory nature of the GA as it seeks the optimal solution.

Feature Selection Across Generations: This heatmap visualizes which features (columns) were selected in the best-performing chromosome of each generation (rows). White indicates a selected feature, and black indicates a non-selected feature.

Feature Selection Frequency: This bar chart shows how many times each feature was selected across all generations, highlighting the most consistently important features.

Conclusion
This project successfully demonstrates that Genetic Algorithms are an effective method for feature selection in high-dimensional datasets. By systematically identifying and selecting the most impactful features, the GA helps to build a more robust and efficient SVM classifier for cancer prediction. This methodology has broad applications in fields where data is complex, including healthcare diagnostics, customer segmentation, and predictive maintenance.

References
Liu, B. G., Xu, L. J., Wang, Y. H., & Tang, J. H. (2012). Genetic Algorithm-Based Feature Selection for Classification: A Comparative Study. IEEE International Conference on Systems, Man, and Cybernetics (SMC).

García, J. S., Gómez, J. A., & Carrillo, A. J. L. (2013). Feature Selection Using Genetic Algorithms: A Review. IEEE Transactions on Evolutionary Computation.

Pal, S. K., Sinha, S. K., & Chaudhuri, B. B. (2011). Feature Selection for Classification Using Genetic Algorithms. Pattern Recog---
