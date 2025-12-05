import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the training data
training_data = np.loadtxt('handwriting_training_set.txt', delimiter=',')
training_labels = np.loadtxt('handwriting_training_set_labels.txt')

# Load the test data
test_data = np.loadtxt('handwriting_test_set.txt', delimiter=',')
test_labels = np.loadtxt('handwriting_test_set_labels_for_Python.txt')

# Separate training data by digit class
digit_matrices = {}
svd_results = {}

print("\nSeparating training data by digit class")
for digit in range(10):
    # Find indices for this digit
    if digit == 0:
        # Digit 0 is labeled as 10
        indices = np.where(training_labels == 10)[0]
    else:
        indices = np.where(training_labels == digit)[0]
    
    # Extract the training examples for this digit
    digit_matrix = training_data[indices, :]
    digit_matrices[digit] = digit_matrix
    print(f"Digit {digit}: {digit_matrix.shape[0]} examples")

# Compute SVD for each digit class
print("\nComputing SVD for each digit class")
for digit in range(10):
    U, s, Vt = np.linalg.svd(digit_matrices[digit], full_matrices=False)
    svd_results[digit] = {'U': U, 's': s, 'Vt': Vt}
    print(f"Digit {digit}: SVD computed, {len(s)} singular values")

def classify_digit(test_vector, k):
    # Classify by finding which digit's subspace the test vector fits best
    best_digit = 0
    minimumDistance = float('inf')
    
    for digit in range(10):
        # Get the top k singular vectors as basis for this digit
        V_k = svd_results[digit]['Vt'][:k, :].T  
        
        # Project test vector onto this digit's subspace
        projection = V_k @ (V_k.T @ test_vector)
        
        # Distance from test vector to projection (smaller = better fit)
        distance = np.linalg.norm(test_vector - projection)
        
        if distance < minimumDistance:
            minimumDistance = distance
            best_digit = digit
    
    return best_digit

# Classify test digits using different numbers of singular vectors
k_values = [5, 10, 15, 20]

print("\nClassifying test digits")

# Store accuracy results
accuracy_results = []

for k in k_values:
    print(f"\nUsing {k} singular vectors:")
    
    correct = 0
    total = len(test_data)
    
    # Classify each test digit
    for i in range(total):
        test_digit = test_data[i, :]
        predicted = classify_digit(test_digit, k)
        
        # Get true label
        true_label = int(test_labels[i])
        if true_label == 10:
            true_label = 0
        
        if predicted == true_label:
            correct += 1
    
    accuracy = (correct / total) * 100
    accuracy_results.append(accuracy)
    print(f"Accuracy: {accuracy:.2f}%")

# Print table
print("\nAccuracy Results:")
print("Number of Basis Vectors | Accuracy (%)")
print("--------------------------------------")
for k, acc in zip(k_values, accuracy_results):
    print(f"         {k}            |    {acc:.2f}%")

# Problem A | Task i
# Create graph
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracy_results, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Basis Vectors', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Classification Accuracy vs Number of Basis Vectors', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.ylim([0, 100])
for k, acc in zip(k_values, accuracy_results):
    plt.text(k, acc + 2, f'{acc:.2f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Problem B

digit_limit = 1.2

def classify_two_stage_digit(test_vector):

    k_stage_1 = 0
    distances_b = {}

    for digit in range(10):
        
        VectorOne = svd_results[digit]['Vt'][k_stage_1, :].reshape(-1, 1)
        reshaped_test_vector = test_vector.reshape(-1, 1)
        proj_stage1 = VectorOne @ (VectorOne.T @ reshaped_test_vector)
        distance_b = np.linalg.norm(reshaped_test_vector - proj_stage1)
        distances_b[digit] = distance_b

    distance_values = list(distances_b.items())
    starting_values = distance_values[0]
    best_digit = starting_values[0]
    minimumDistance = starting_values[1]

    second_minDistance = minimumDistance
    
    for digit, distance in  distance_values[1:]:

        if distance < minimumDistance:

            # Store old value
            second_minDistance = minimumDistance
            
            # Store new values
            minimumDistance = distance
            best_digit = digit
            
        elif distance < second_minDistance:

            second_minDistance = distance

    if minimumDistance * digit_limit < second_minDistance:

        return best_digit, 1 

    
    if accuracy_results:

        max_accuracy = max(accuracy_results)
        max_accuracy_index = accuracy_results.index(max_accuracy)
        max_k_value = k_values[max_accuracy_index]

    else:
      
        print(f"We are unable to get max_value from Accuracy Result Stage 2")
        SystemExit

    print(f"{max_k_value} is the Highest Value Of K from Problem 1 This will be used for Stage 2")

    predicted_stage_2 = classify_digit(test_vector, max_k_value)
    return predicted_stage_2, 2

# Run Stage Two
print(f"\n========================================================================================")#
print(f"\nRunning Two-Stage Algorithm | Problem B\n")

correct_b = 0
total_b = len(test_data)
stage_one_count = 0
stage_two_count = 0

# Same as Problem 1, but we have 
for i in range(total_b):
    test_digit = test_data[i, :]
    predicted, stageCheck = classify_two_stage_digit(test_digit)
    
    # Track which stage was used
    if stageCheck == 1:
        stage_one_count += 1
    elif stageCheck == 2:
        stage_two_count += 1
    
    true_label = int(test_labels[i])
    if true_label == 10:
        true_label = 0
    
    if predicted == true_label:
        correct_b += 1

accuracy_b = (correct_b / total_b) * 100
percent_stage_one = (stage_one_count / total_b) * 100
percent_stage_two = (stage_two_count / total_b) * 100

print("Problem B Results")
print(f"|Total Correct (Fraction)|")
print(f"      {correct_b}/{total_b}         ")
print(f"|Total Correct (Percentage)|")
print(f"      {accuracy_b:.2f}%")

print(f"\n| Stage 1 Uses |")
print(f"      {stage_one_count}         ")
print(f"|Stage 1 Percentages|")
print(f"      {percent_stage_one:.2f}%         ")

print(f"\n| Stage 2 Uses |")
print(f"      {stage_two_count}         ")
print(f"|Stage 2 Percentages|")
print(f"      {percent_stage_two:.2f}%         ")


# "Is it possible to get as good a result for this version?" From Instructions
# Get the highest accuracy from Problem 1A
if accuracy_results:
    bestAccuracy_a = max(accuracy_results)
else:
    bestAccuracy_a = 0

print("\nQuestions")
print(f"Best Accuracy from Problem A: {bestAccuracy_a:.2f}%")
print("\nIs it possible to get as good a result for this version? From Instructions")

if accuracy_b >= bestAccuracy_a:
    print(f"Yes Problem B had a percentage {accuracy_b:.2f}% had a higher or equal accuracy than Problem 1A which had an accuracy {bestAccuracy_a:.2f}%")
elif bestAccuracy_a > accuracy_b:
    print(f"No Problem B had a percentage {accuracy_b:.2f}% was is lower accuracy to Problem 1A which had an accuracy of {bestAccuracy_a:.2f}%).")

#"How frequently is the second stage necessary? From Instructions"
print(f"\nHow frequently is the second stage necessary? From Instructions")
print(f"Stage Two was necessary for {stage_two_count} Stages out of {total_b} Test Examples")

