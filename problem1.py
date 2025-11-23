import numpy as np

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
    min_distance = float('inf')
    
    for digit in range(10):
        # Get the top k singular vectors as basis for this digit
        V_k = svd_results[digit]['Vt'][:k, :].T  # Shape: (400, k)
        
        # Project test vector onto this digit's subspace
        projection = V_k @ (V_k.T @ test_vector)
        
        # Distance from test vector to projection (smaller = better fit)
        distance = np.linalg.norm(test_vector - projection)
        
        if distance < min_distance:
            min_distance = distance
            best_digit = digit
    
    return best_digit

# Classify test digits using different numbers of singular vectors
k_values = [5, 10, 15, 20]

print("\nClassifying test digits")

for k in k_values:
    print(f"\nUsing {k} singular vectors:")
    
    # Classify each test digit
    for i in range(len(test_data)):
        test_digit = test_data[i, :]
        predicted = classify_digit(test_digit, k)
        # Classifications are computed and stored
    
    print(f"Classified {len(test_data)} test digits")

