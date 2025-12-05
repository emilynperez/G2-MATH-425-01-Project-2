# Part II: per-digit accuracy and misclassified examples

Xtrain = load("handwriting_training_set.txt");
ytrain = load("handwriting_training_set_labels.txt");
Xtest  = load("handwriting_test_set.txt");
ytest  = load("handwriting_test_set_labels_for_Python.txt");
# y_pred must be produced by your classifier.
# Option 1: it was saved to a file y_pred.mat
load("y_pred.mat");   % this file should contain a vector y_pred

if ~exist("y_pred","var")
    error("y_pred not found. Run the classifier first and save y_pred.mat.");
end

digits_labels = sort(unique(ytest));
n_digits = numel(digits_labels);
acc_per_digit = zeros(n_digits,1);

for k = 1:n_digits
    d = digits_labels(k);
    idx = (ytest == d);
    n_k = sum(idx);
    if n_k == 0
        acc_per_digit(k) = NaN;
    else
        acc_per_digit(k) = 100 * sum(y_pred(idx) == d) / n_k;
    end
end

T = table(digits_labels, acc_per_digit, ...
          'VariableNames', {'Digit','AccuracyPercent'});
disp(T);

figure;
bar(digits_labels, acc_per_digit);
xlabel("Digit");
ylabel("Accuracy (%)");
title("Per-digit classification accuracy");

wrongIdx = find(y_pred ~= ytest);
fprintf("Total misclassified: %d out of %d\n", numel(wrongIdx), numel(ytest));

num_to_show = min(9, numel(wrongIdx));
exampleIdx = wrongIdx(1:num_to_show);

figure;
for i = 1:num_to_show
    idx = exampleIdx(i);
    img = reshape(Xtest(idx,:), 20, 20);
    subplot(3,3,i);
    imagesc(img');
    colormap gray;
    axis off;
    title(sprintf("True %d, Pred %d", ytest(idx), y_pred(idx)));
end
sgtitle("Misclassified test digits");
