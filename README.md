# 🤖 DeepCSAT — Customer Satisfaction Prediction using ANN 💬📊

A comprehensive **Artificial Neural Network (ANN)**–based project designed to **predict Customer Satisfaction (CSAT) scores** using a blend of numerical, categorical, and textual review data.  
DeepCSAT empowers organizations to **understand, analyze, and improve customer experiences** through predictive analytics and sentiment insights.

---

## 🎯 Objective

- Predict customer satisfaction (CSAT) scores based on historical order and review data.  
- Analyze textual sentiments and numerical attributes (order value, response time, etc.).  
- Provide actionable insights to enhance customer service and experience.

---

## 🔧 Features

### 🧮 Data Processing
- Handles **numerical, categorical, and text** features simultaneously.  
- Text cleaned using **NLTK** (stopword removal, tokenization).  
- Dimensionality reduction using **TruncatedSVD** for text embeddings.

### 🧠 Model Architecture
- Built using **TensorFlow / Keras Sequential API**.  
- Layers:
  - Dense(32, ReLU) + Dropout(0.3)  
  - Dense(16, ReLU)  
  - Dense(1, Linear Output)  
- Optimized with **Adam** and **Early Stopping**.

### ⚙️ Preprocessing Pipeline
- **Numeric**: Imputation + Standard Scaling  
- **Categorical**: OneHot Encoding  
- **Text**: TF-IDF + SVD for feature compression  
- Modular pipeline built using **scikit-learn’s ColumnTransformer**.

### 📈 Evaluation Metrics
- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  
- **R² Score**  
- Visualization of Predicted vs Actual CSAT scores.

---

## 🧠 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python |
| ML Framework | TensorFlow / Keras |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Text Analysis | NLTK, TF-IDF, TruncatedSVD |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib, H5 |

---

## 📁 Project Structure

```

DeepCSAT/
├── data/
│   └── clean_sample.csv                # Input dataset
├── models/
│   ├── preprocessor.joblib             # Saved preprocessing pipeline
│   ├── best_model.h5                   # Best ANN model checkpoint
│   └── final_model_saved/              # Final trained model
├── notebooks/
│   └── DeepCSAT_Training_Notebook.ipynb  # Full model development
├── src/
│   ├── preprocess.py                   # Text cleaning & data prep
│   ├── train_model.py                  # ANN training script
│   └── evaluate.py                     # Model evaluation & visualization
├── requirements.txt                    # Dependencies list
└── README.md                           # Project documentation

````

---

## 🧪 Example Workflow

1. **Load and Clean Data**
   ```python
   df = pd.read_csv('data/clean_sample.csv')
   df['review_text_clean'] = df['review_text'].apply(clean_text)
````

2. **Build and Train ANN**

   ```python
   model = build_model(input_dim=X_train.shape[1])
   model.fit(X_train, y_train, validation_split=0.15, epochs=20, batch_size=32)
   ```

3. **Evaluate**

   ```python
   y_pred = model.predict(X_test)
   print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
   ```

4. **Predict New Review**

   ```python
   sample_input = X_test[:1]
   predicted_csat = model.predict(sample_input)[0][0]
   print('Predicted CSAT Score:', round(predicted_csat, 2))
   ```

---

## 📊 Example Insights

| Metric   | Value |
| -------- | ----- |
| RMSE     | 0.48  |
| MAE      | 0.35  |
| R² Score | 0.87  |

**Interpretation:**
The model demonstrates strong predictive accuracy and generalization.
Organizations can identify service patterns, prioritize improvements, and predict customer satisfaction trends.

---

## 💡 Business Impact

✅ **Customer Retention:** Predict dissatisfaction early to take proactive measures.
✅ **Operational Insights:** Detect service areas that drive satisfaction/dissatisfaction.
✅ **Experience Enhancement:** Design better user experiences through data-driven understanding.
✅ **Scalable Deployment:** ANN pipeline ready for integration into dashboards or APIs.

---

## 🙋‍♀️ Created By

**Sandhiya Sree V**
📧 [sandhiyasreev@gmail.com](mailto:sandhiyasreev@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/sandhiya-sree-v-3a2321298/)
🌐 [GitHub](https://github.com/Sandhiyasreev)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share with credit.

⭐ If you found this project helpful, give it a **star** on GitHub!
💬 Feedback and collaborations are always welcome.

```
