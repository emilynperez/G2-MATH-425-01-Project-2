# -*- coding: utf-8 -*-


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
    
# Compute SVD for each digit class
print("\nComputing SVD for each digit class")
for digit in range(10):
    U, s, Vt = np.linalg.svd(digit_matrices[digit], full_matrices=False)
    svd_results[digit] = {'U': U, 's': s, 'Vt': Vt}
   


# Classify test digits using different numbers of singular vectors
k_values = [5, 10, 15, 20]

#--------------------------------------------------------------------
#part 3
print("-"*60)
print("\npart 3: \n-singular values analisis \n-analizing singular values for each digit class")

# to store the cumulate variance array for each digit
cum_variance_data = {}


for digit in range(10):
    # retrieves the singular values 
    s = svd_results[digit]['s']
    # calculating the total variance
    total_variance = np.sum(s**2)
    #compute the running total of the square singular values and divide to total variance
    normalized_cumulative_variance = np.cumsum(s**2)/total_variance
    # store the resulting 400_elements array of cumulative (range from 0 to 1)
    cum_variance_data[digit] = normalized_cumulative_variance
    
# print the table of cumulative variance up to k = 20

# hold the data for the final table columns
cum_variance_table = {k: [] for k in k_values}

# loops through the target basis vector counts k_values (5,10,15,20)
for k in k_values:
    for digit in range(10):
        # Retrieve the cumulative variance percentage at that k
        # k is 1-index, array is 0-index, so index k-1
        
        cum_variance_table[k].append(f"{cum_variance_data[digit][k-1]*100:.2f}%")
#create a pandas dataframe from the cumulative variance table
cum_variance_df = pd.DataFrame(
    cum_variance_table,
    index=[f'Digit {d}' for d in range(10)])
#assign columns names to the dataframe
cum_variance_df.columns = [f'k={k}' for k in k_values]
print("\nCumulative Explained Variance by k:")


# print the table 
print(cum_variance_df.to_markdown())

# plotting the visual confirmation of diffent decay rates
plt.figure(figsize=(15,8))
for digit in range(10):
    plt.plot(range(1,31), cum_variance_data[digit][:30], label=f'Digit{digit}')
    
    
plt.title("cumulative explained variance vs number of basis vectors (k)")
plt.xlabel('number of basis vector (k)')
plt.ylabel('cumulative expplain variance')
plt.grid(True)
plt.legend(ncol=5,loc='lower right')
plt.xlim(1,30)
plt.show()
