import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Elastic Net Model Presentation")

# Introduction
st.header("1. Introduction to Elastic Net Model")
st.write("""
- **What is Elastic Net?** Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalties of the Lasso and Ridge methods.
- **Why use Elastic Net?** It addresses some limitations of Lasso (like selecting only one feature from a group of correlated features) and Ridge (like not performing feature selection).
""")

# Add an image related to Elastic Net
st.image("/Users/kh/Downloads/Tuwaiq_BootCamp/Others/Lightning-talk/Elastic Net Illustration.webp", caption="Elastic Net Illustration")

# Mathematical Background
st.header("2. Mathematical Background")
st.write("""
- **Lasso Regression (L1 Regularization)**: Adds a penalty equal to the absolute value of the magnitude of coefficients.
- **Ridge Regression (L2 Regularization)**: Adds a penalty equal to the square of the magnitude of coefficients.
- **Elastic Net Combination**: Combines both L1 and L2 regularization.
""")

# Advantages
st.header("3. Advantages of Elastic Net")
st.write("""
- **Handling Multicollinearity**: Can handle correlated features better than Lasso.
- **Feature Selection**: Can select a group of correlated features.
- **Regularization**: Provides a balance between L1 and L2 regularization.
""")

# Add a chart to illustrate advantages
advantages = pd.DataFrame({
    'Method': ['Lasso', 'Ridge', 'Elastic Net'],
    'Feature Selection': [1, 0, 1],
    'Handling Multicollinearity': [0, 1, 1]
})
st.bar_chart(advantages.set_index('Method'))

# Elastic Net in Action
st.header("4. Elastic Net in Action")

# Generate dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Interactive sliders for alpha and l1_ratio
alpha = st.slider("Alpha (Regularization Strength)", 0.01, 1.0, 0.1)
l1_ratio = st.slider("L1 Ratio (Mixing Parameter)", 0.0, 1.0, 0.5)

# Fit the Elastic Net model
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display results
st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Plot true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
st.pyplot(plt)

# Feature Coefficients
st.subheader("Feature Coefficients")
coefficients = pd.DataFrame({
    'Feature': [f'Feature {i}' for i in range(X.shape[1])],
    'Coefficient': model.coef_
})
st.dataframe(coefficients)

# Plot coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Coefficient', data=coefficients)
plt.xticks(rotation=90)
plt.title('Feature Coefficients')
st.pyplot(plt)

# Comparison with Other Models
st.header("5. Comparison with Other Models")
st.write("""
- **Lasso**: Useful for feature selection but can ignore groups of correlated features.
- **Ridge**: Shrinks coefficients but doesn't perform feature selection.
- **Elastic Net**: Combines the strengths of both Lasso and Ridge.
""")

# Use Cases
st.header("6. Use Cases and Applications")
st.write("""
Elastic Net is particularly useful in situations where:
- There are multiple features that are correlated.
- Feature selection is important.
- Balancing bias-variance tradeoff is crucial.
""")

# Add an image for use cases
st.image("/Users/kh/Downloads/Tuwaiq_BootCamp/Others/Lightning-talk/Elastic Net Use Cases.webp", caption="Elastic Net Use Cases")

# Conclusion
st.header("7. Conclusion")
st.write("""
Elastic Net provides a flexible and effective approach to regression modeling by combining the benefits of both Lasso and Ridge regression. Its ability to handle multicollinearity and perform feature selection makes it a powerful tool for various applications.
""")

# Footer
st.write("Thank you for your attention! Feel free to experiment with the parameters and explore the model further.")