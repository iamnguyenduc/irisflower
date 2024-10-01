import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import KFold #Nhom 12B #Nhom 12B #Nhom 12B

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (class labels)

# KNN implementation with customizable K #Nhom 12B #Nhom 12B
def knn_predict(test_sample, X_train, y_train, k=3):
    # Calculate Euclidean distances from the test sample to all training samples
    distances = euclidean_distances([test_sample], X_train)[0]
    
    # Sort the distances and their corresponding labels
    sorted_indices = distances.argsort()
    sorted_labels = y_train[sorted_indices]

    # Get the labels of the k-nearest neighbors
    nearest_neighbors_labels = sorted_labels[:k]
    
    # Find the majority class among the k-nearest neighbors
    unique, counts = np.unique(nearest_neighbors_labels, return_counts=True)
    majority_class = unique[counts.argmax()]

    return majority_class
 
# K-fold cross-validation #Nhom 12B #Nhom 12B #Nhom 12B
def k_fold_validation(X, y, k=5, knn_k=3):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Predictions for the test set
        y_pred = []
        
        for test_sample in X_test:
            predicted_class = knn_predict(test_sample, X_train, y_train, knn_k)
            y_pred.append(predicted_class)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return np.mean(accuracies), accuracies

# Main program to evaluate using K-fold cross-validation
def main():
    print("=== ĐÁNH GIÁ MÔ HÌNH KNN BẰNG PHƯƠNG PHÁP K-FOLD CROSS-VALIDATION ===")
    
    k_value = int(input("Nhập giá trị K cho KNN (K = 5): "))
    while k_value not in [5]:
        print("K phải bằng 5. Vui lòng nhập lại.")
        k_value = int(input("Nhập giá trị K cho KNN (K = 5): "))

    mean_accuracy, accuracies = k_fold_validation(X, y, k=5, knn_k=k_value)
    
    # Display results
    print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
    print(f"Độ chính xác trung bình của mô hình với K={k_value}: {mean_accuracy:.2f}")
    print("\n=== ĐỘ CHÍNH XÁC THEO TỪNG LẦN LẶP ===")
    for i, acc in enumerate(accuracies, start=1):
        print(f"Lần lặp {i}: Độ chính xác = {acc:.2f}")

# Run the program
main()
