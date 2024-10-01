import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import euclidean_distances #Nhom 12B #Nhom 12B #Nhom 12B #Nhom 12B

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (class labels)

# KNN implementation with customizable K #Nhom 12B #Nhom 12B #Nhom 12B
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

# Leave-One-Out Cross-Validation (LOOCV) #Nhom 12B #Nhom 12B #Nhom 12B
def loocv(k=3):
    n_samples = X.shape[0]
    y_pred = []

    for i in range(n_samples):
        # Prepare training and test sets
        X_train = np.delete(X, i, axis=0)  # All samples except the ith
        y_train = np.delete(y, i, axis=0)
        test_sample = X[i]  # The ith sample
        
        # Predict the class of the test sample
        predicted_class = knn_predict(test_sample, X_train, y_train, k)
        y_pred.append(predicted_class)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    return accuracy, y_pred

# Main program to evaluate using LOOCV
def main():
    print("=== ĐÁNH GIÁ MÔ HÌNH KNN BẰNG PHƯƠNG PHÁP LOOCV ===")
    
    k_value = int(input("Nhập giá trị K (3 hoặc 5): "))
    while k_value not in [3, 5]:
        print("K phải bằng 3 hoặc 5. Vui lòng nhập lại.")
        k_value = int(input("Nhập giá trị K (3 hoặc 5): "))

    accuracy, y_pred = loocv(k=k_value)
    
    # Display results
    print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
    print(f"Độ chính xác của mô hình với K={k_value}: {accuracy:.2f}")
    print("\n=== NHÃN DỰ ĐOÁN ===")
    for i, predicted in enumerate(y_pred):
        print(f"Mẫu thứ {i + 1}: Nhãn dự đoán: {predicted}")

# Run the program
main()
