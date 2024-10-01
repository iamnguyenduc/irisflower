import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import euclidean_distances #Nhom 12B #Nhom 12B #Nhom 12B #Nhom 12B #Nhom 12B

# Tải dữ liệu hoa iris
iris = load_iris()
X = iris.data  # Đặc trưng
y = iris.target  # Class

# Chia dữ liệu thành tập test, tập train
def get_train_test_data(test_indices):
    test_indices = [i - 1 for i in test_indices]
    X_test = X[test_indices]  
    y_test = y[test_indices]
    
    X_train = np.delete(X, test_indices, axis=0)
    y_train = np.delete(y, test_indices, axis=0)
    
    return X_train, y_train, X_test, y_test

# Thực hiện KNN #Nhom 12B #Nhom 12B #Nhom 12B
def knn_predict(test_sample, X_train, y_train, k=3, test_index=None):
    # Tính khoảng cách Euclidean từ mẫu test đến tập train
    distances = euclidean_distances([test_sample], X_train)[0]
    
    # Sắp xếp khoảng cách tăng dần và nhãn tương ứng
    sorted_indices = distances.argsort()
    sorted_distances = distances[sorted_indices]
    sorted_labels = y_train[sorted_indices]

    print("Sắp xếp khoảng cách và nhãn tương ứng:")
    for i, train_index in enumerate(sorted_indices):
        print(f"d({test_index},{train_index + 1}) = {sorted_distances[i]:.4f}, nhãn: {sorted_labels[i]}")

    nearest_neighbors_labels = sorted_labels[:k]             # K nhãn của K khoảng cách dmin
    
    unique, counts = np.unique(nearest_neighbors_labels, return_counts=True)
    majority_class = unique[counts.argmax()]                 # tìm nhãn chiếm ưu thế

    return majority_class

# Function to run the KNN with input test indices and K value
def run_knn(test_indices, k=3):
    X_train, y_train, X_test, y_test = get_train_test_data(test_indices)
    
    for i, test_sample in enumerate(X_test):
        print(f"\nMẫu test {test_indices[i]}:")
        predicted_class = knn_predict(test_sample, X_train, y_train, k, test_indices[i])
        print(f"Nhãn dự đoán của mẫu test {test_indices[i]}: {predicted_class}")
        print(f"Nhãn thực tế của mẫu test {test_indices[i]}: {y_test[i]}")
        print()

# Main program to ask for user input
def main():
    # Get test sample indices from user input
    test_indices = input("Nhập mẫu test (e.g., 1, 51, 101): ")
    test_indices = list(map(int, test_indices.split(',')))  # Convert to a list of integers

    # Get the value of K from user input (only 3 or 5 allowed)
    k_value = int(input("Nhập giá trị K (3 or 5): "))
    while k_value not in [3, 5]:
        print("K phải bằng 3 hoặc 5.")
        k_value = int(input("Nhập giá trị K (3 or 5): "))

    # Run the KNN algorithm with user inputs
    run_knn(test_indices, k_value)

# Run the program
main()
