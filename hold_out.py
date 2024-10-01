import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import euclidean_distances #Nhom 12B #Nhom 12B #Nhom 12B

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

# Hold-out cross-validation (30:70) #Nhom 12B #Nhom 12B #Nhom 12B
def hold_out_validation(test_size=0.3, k=3):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Predictions for the test set
    y_pred = []
    
    for test_sample in X_test:
        predicted_class = knn_predict(test_sample, X_train, y_train, k)
        y_pred.append(predicted_class)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred

# Main program to evaluate using hold-out cross-validation
def main():
    print("=== ĐÁNH GIÁ MÔ HÌNH KNN BẰNG PHƯƠNG PHÁP HOLD-OUT CROSS-VALIDATION ===")
    
    k_value = int(input("Nhập giá trị K (3 hoặc 5): "))
    while k_value not in [3, 5]:
        print("K phải bằng 3 hoặc 5. Vui lòng nhập lại.")
        k_value = int(input("Nhập giá trị K (3 hoặc 5): "))

    accuracy, y_test, y_pred = hold_out_validation(k=k_value)
    
    # Display results
    print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
    print(f"Độ chính xác của mô hình với K={k_value}: {accuracy:.2f}")
    print("\n=== NHÃN THỰC TẾ VÀ NHÃN DỰ ĐOÁN ===")
    for index, (actual, predicted) in enumerate(zip(y_test, y_pred), start=1):
        print(f"Mẫu số {index}: Nhãn thực tế: {actual}, Nhãn dự đoán: {predicted}")

# Run the program
main()
