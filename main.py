import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearRegressionScratch:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.losses = []
        self.m = 0
        self.b = 0

    def predict(self, X):
        return self.m * X + self.b

    def fit(self, X, y):
        n = len(X)
        
        for i in range(self.epochs):
            y_pred = self.predict(X)
            
            loss = np.mean((y - y_pred)**2)
            self.losses.append(loss)
            
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            
            self.m -= self.lr * dm
            self.b -= self.lr * db
            
            if i % 100 == 0:
                print(f"epoch: {i}  loss: {loss}")
            
    def calculate_r2(self, y_true, y_pred):
        SSR = sum((y_true - y_pred)**2)
        SST = sum((y_true - np.mean(y_true))**2)
        
        r2 = 1 - (SSR/SST)
        return r2
    
def main():
    X = np.random.normal(5, 2, 100)
    y = 5 * X + 30 + np.random.normal(0, 3, 100)
    
    model = SimpleLinearRegressionScratch()
    model.fit(X,y)
    
    predictions = model.predict(X)
    r2 = model.calculate_r2(y, predictions)
    
    print(f"\nFinal Equation: y = {model.m:.2f}x + {model.b:.2f}")
    print(f"Model Accuracy (R2 Score): {r2:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


    ax1.scatter(X, y, color='blue', label='Actual Data', alpha=0.5)
    ax1.plot(X, predictions, color='red', label='Regression Line', linewidth=2)
    ax1.set_title("Study Hours vs Score")
    ax1.legend()

    ax2.plot(range(len(model.losses)), model.losses, color='orange')
    ax2.set_title("Loss Reduction over Epochs (Gradient Descent)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("MSE Loss")
    
    plt.show()
    
if __name__ == "__main__":
    main()

