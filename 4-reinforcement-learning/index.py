import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ======================
# 1. Load Dataset
# ======================
iris = load_iris()
X = iris.data          # fitur
y = iris.target.reshape(-1, 1)  # label

# One-hot encoding label
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 2. Inisialisasi Bobot
# ======================
input_size = 4
hidden_size = 10
output_size = 3

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 1000

# ======================
# 3. Fungsi Aktivasi
# ======================
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return x > 0

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ======================
# 4. Training
# ======================
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    output = softmax(z2)

    # Loss (cross-entropy)
    loss = -np.mean(np.sum(y_train * np.log(output + 1e-8), axis=1))

    # Backpropagation
    d_output = output - y_train
    dW2 = np.dot(a1.T, d_output)
    db2 = np.sum(d_output, axis=0, keepdims=True)

    da1 = np.dot(d_output, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update bobot
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ======================
# 5. Testing
# ======================
z1 = np.dot(X_test, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
output = softmax(z2)

prediksi = np.argmax(output, axis=1)
label_asli = np.argmax(y_test, axis=1)

akurasi = np.mean(prediksi == label_asli)
print("Akurasi:", akurasi)
