# 📉 Linear Regression From Scratch

A deep-dive into the foundational mathematics of Machine Learning. This project implements a **Simple Linear Regression** model using only **NumPy**, without the help of high-level libraries like Scikit-Learn.

## 🧠 The Math Behind the Code

This implementation bridges the three pillars of Machine Learning math:

### 1. Linear Algebra (Data Representation)
The relationship between the features ($X$) and the target ($y$) is represented by the linear equation:
$$y = mx + b$$
Where **m** is the weight (slope) and **b** is the bias (intercept).

### 2. Calculus (Gradient Descent)
To find the optimal values for $m$ and $b$, I implemented **Gradient Descent**. The model calculates the Partial Derivatives of the Mean Squared Error (MSE) to determine the direction and magnitude of the update:
- $\frac{\partial}{\partial m} = \frac{-2}{n} \sum X(y - y_{pred})$
- $\frac{\partial}{\partial b} = \frac{-2}{n} \sum (y - y_{pred})$

### 3. Statistics (Evaluation)
The model's performance is measured using the **R-Squared ($R^2$)** score, which indicates how much of the variance in the target variable is explained by the features:
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

## 🚀 Features
- **Custom Fit/Predict API:** Mimics the Scikit-Learn structure for ease of use.
- **Dynamic Loss Tracking:** Monitors the Mean Squared Error across training epochs.
- **Visualization:** Includes automated plots for the regression line and the gradient descent loss curve.

## 📂 Project Structure
- `main.py`: The Python class and execution logic.
- `README.md`: Documentation of the mathematical concepts.

## 🛠️ Usage
```bash
    python main.py
```

---
This project marks the completion of the **Math Phase** of my **ML Journey**. It demonstrates a fundamental understanding of how models learn through optimization.
