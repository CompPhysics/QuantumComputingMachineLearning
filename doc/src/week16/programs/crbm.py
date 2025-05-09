import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights = np.random.normal(0, 0.01, size=(n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sample_prob(self, probs):
        return (np.random.rand(*probs.shape) < probs).astype(np.float32)

    def train(self, data, epochs=1000, batch_size=10):
        n_samples = data.shape[0]

        for epoch in range(epochs):
            np.random.shuffle(data)
            for i in range(0, n_samples, batch_size):
                v0 = data[i:i + batch_size]
                # Positive phase
                h0_prob = self.sigmoid(np.dot(v0, self.weights) + self.hidden_bias)
                h0_sample = self.sample_prob(h0_prob)

                # Negative phase
                v1_prob = self.sigmoid(np.dot(h0_sample, self.weights.T) + self.visible_bias)
                h1_prob = self.sigmoid(np.dot(v1_prob, self.weights) + self.hidden_bias)

                # Update weights and biases
                self.weights += self.learning_rate * (
                    np.dot(v0.T, h0_prob) - np.dot(v1_prob.T, h1_prob)
                ) / batch_size
                self.visible_bias += self.learning_rate * np.mean(v0 - v1_prob, axis=0)
                self.hidden_bias += self.learning_rate * np.mean(h0_prob - h1_prob, axis=0)

            if epoch % 100 == 0:
                error = np.mean((v0 - v1_prob) ** 2)
                print(f"Epoch {epoch}: Reconstruction error = {error:.4f}")

    def transform(self, v):
        """Compute hidden unit probabilities."""
        return self.sigmoid(np.dot(v, self.weights) + self.hidden_bias)

    def reconstruct(self, v):
        """Reconstruct visible units from hidden layer."""
        h = self.sigmoid(np.dot(v, self.weights) + self.hidden_bias)
        v_recon = self.sigmoid(np.dot(h, self.weights.T) + self.visible_bias)
        return v_recon

# Generate synthetic binary data
data = np.random.randint(0, 2, size=(100, 6))

# Initialize and train RBM
rbm = RBM(n_visible=6, n_hidden=2, learning_rate=0.1)
rbm.train(data, epochs=500)

# Transform and reconstruct
sample = np.array([[1, 0, 1, 0, 1, 0]])
hidden = rbm.transform(sample)
reconstructed = rbm.reconstruct(sample)

print("Original:", sample)
print("Hidden:", hidden)
print("Reconstructed:", reconstructed)
