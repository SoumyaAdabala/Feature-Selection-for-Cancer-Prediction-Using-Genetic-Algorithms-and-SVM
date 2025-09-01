% Load the saved data from the file
load("C:\Users\soumy\Desktop\IC_Assignment\Ic_Presentation\cancer_data.mat", 'X', 'y');

% Ensure the dataset 'X' and 'y' are available after loading
[nSamples, nFeatures] = size(X);
if nFeatures ~= 30
    error('The dataset does not have 30 features. Please check the data.');
end

% Define GA parameters
populationSize = 50;
numGenerations = 50;
mutationRate = 0.05;
numFeaturesToSelect = ceil(0.5 * nFeatures); % Target is to select half the features

% Fitness function: we use classification accuracy as the fitness score
%The for loop runs through each of the 5 folds of cross-validation.
%trainIdx and testIdx are indices that indicate which samples are used for training and testing in each fold.
%The fitcsvm function trains a Support Vector Machine (SVM) classifier using the selected features (selectedX) and the corresponding labels (y).
%The trained SVM model is used to make predictions on the test set (predict(model, selectedX(testIdx, :))).
%The accuracy for each fold is calculated by comparing the predictions to the actual labels (y(testIdx)).
function fitness = fitnessFunction(selectedFeatures, X, y)
    % Select the features according to the GA's binary chromosome
    selectedX = X(:, logical(selectedFeatures));
    
    % Train a classifier ( SVM)
    if isempty(selectedX) % Handle cases where no features are selected
        fitness = 0;
        return;
    end
     
    % Use k-fold cross-validation to evaluate model performance Support Vector Machine
    cv = cvpartition(y, 'KFold', 5);
    accuracy = zeros(cv.NumTestSets, 1);
    
    for i = 1:cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        
        model = fitcsvm(selectedX(trainIdx, :), y(trainIdx)); % SVM model
        predictions = predict(model, selectedX(testIdx, :));
        accuracy(i) = sum(predictions == y(testIdx)) / length(y(testIdx));
    end
    
    % Fitness is the mean accuracy across the folds
    fitness = mean(accuracy);
end

% Initialize population (binary chromosomes)
population = randi([0 1], populationSize, nFeatures);

% Initialize arrays to store best fitness and feature selections over generations
bestFitnessOverGenerations = zeros(numGenerations, 1);
bestIndividualsOverGenerations = zeros(numGenerations, nFeatures);

% Evolutionary process: Generations loop
for generation = 1:numGenerations
    fitnessScores = zeros(populationSize, 1);
    for i = 1:populationSize
        fitnessScores(i) = fitnessFunction(population(i, :), X, y);
    end
    
    % Selection: Rank-based selection
    [~, sortedIdx] = sort(fitnessScores, 'descend');
    population = population(sortedIdx, :);
    
    % Save the best individual (feature selection) for this generation
    bestIndividualsOverGenerations(generation, :) = population(1, :);
    
    % Crossover: One-point crossover between the top individuals
    newPopulation = population;
    for i = 1:2:populationSize-1
        crossoverPoint = randi([1, nFeatures-1]);
        newPopulation(i, crossoverPoint+1:end) = population(i+1, crossoverPoint+1:end);
        newPopulation(i+1, crossoverPoint+1:end) = population(i, crossoverPoint+1:end);
    end
    
    % Mutation: Randomly flip bits with a small probability
    mutationMask = rand(populationSize, nFeatures) < mutationRate;
    newPopulation = xor(newPopulation, mutationMask);
    
    % Replace the old population with the new one
    population = newPopulation;
    
    % Store the best fitness score for each generation
    bestFitnessOverGenerations(generation) = max(fitnessScores);
    
    % Display the best fitness score for each generation
    disp(['Generation ', num2str(generation), ': Best Fitness = ', num2str(max(fitnessScores))]);
end

% The best solution after all generations
bestIndividual = population(1, :);
bestFeatures = find(bestIndividual); % Indices of the selected features
disp('Selected Features:');
disp(bestFeatures);

% Train final model using the selected features
finalModel = fitcsvm(X(:, bestFeatures), y);
disp('Final model trained with selected features.');

% Calculate the feature selection frequency
featureSelectionFrequency = sum(bestIndividualsOverGenerations, 1);

% Visualize results
figure;

% Plot 1: Fitness Score Over Generations
subplot(1, 3, 1); % 1 row, 3 columns, position 1
plot(1:numGenerations, bestFitnessOverGenerations, 'LineWidth', 2);
xlabel('Generations');
ylabel('Best Fitness Score');
title('Fitness Score Over Generations');
grid on;

% Plot 2: Heatmap of Feature Selection Across Generations
subplot(1, 3, 2); % 1 row, 3 columns, position 2
imagesc(bestIndividualsOverGenerations); % Each row is a generation, each column a feature
colormap(gray);
colorbar;
xlabel('Features');
ylabel('Generations');
title('Feature Selection Across Generations');

% Plot 3: Bar Plot of Feature Selection Frequency
subplot(1, 3, 3); % 1 row, 3 columns, position 3
bar(1:nFeatures, featureSelectionFrequency, 'FaceColor', [0.2, 0.6, 0.5]);
xlabel('Feature Index');
ylabel('Selection Frequency');
title('Feature Selection Frequency Across Generations');
grid on;
