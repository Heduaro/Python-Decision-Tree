from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the iris dataset
data = load_iris()
x = data.data
y = data.target

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Train a Decision Tree Classifier on the training data
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Make predictions on the training and test data
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

# Calculate the accuracy of the predictions
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f'Train accuracy: {train_acc:.2f}')
print(f'Test accuracy: {test_acc:.2f}')

# 8. The result for the train data represents the accuracy of the model on the training data. It measures how well the model is able to correctly predict the labels of the training data.

# 9. using `random_state` parameter in the `train_test_split` method to get consistent results:
test_accs = []
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=i)
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accs.append(test_acc)

mean_test_acc = np.mean(test_accs)
std_test_acc = np.std(test_accs)

print(f'Mean test accuracy: {mean_test_acc:.2f}')
print(f'Standard deviation of test accuracy: {std_test_acc:.2f}')

# 10. Testing split ratios by changing the `test_size` parameter in the `train_test_split` method.
split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
for split_ratio in split_ratios:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio)
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f'Test accuracy with split ratio {split_ratio}: {test_acc:.2f}')

    # 11. Chart showing accuracy score as function
split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
test_accs = []
for split_ratio in split_ratios:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio)
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_test_pred = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accs.append(test_acc)

plt.plot(split_ratios, test_accs)
plt.xlabel('Split ratio')
plt.ylabel('Test accuracy')
plt.show()

# 12. Exemplary tree using plot tree
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)

plt.figure(figsize=(12,12))
plot_tree(clf,filled=True)
plt.show()