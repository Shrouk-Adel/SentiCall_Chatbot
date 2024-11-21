# SentiCall

This project provides an interactive web application that allows users to analyze Arabic text sentiments, interact with a customer dataset chatbot, and visualize sentiment and customer feedback data using advanced machine learning models and visualization tools.

---

## Features

### 🎯 **Sentiment Analysis**
- Upload Arabic audio files to transcribe and analyze sentiments (Positive, Neutral, Negative).
- Supports sentiment analysis using pre-trained BERT-based models (e.g., BERT and MARBERT).

### 🤖 **Customer Dataset Chatbot**
- Interact with a chatbot powered by Google Generative AI.
- Answer customer-related queries and provide insights from customer feedback data.

### 📊 **Data Visualization**
- Generate charts and graphs, including class distributions and feedback trends.
- Visualize company-specific feedback in interactive formats.

### 🌍 **Tableau Dashboard Integration**
- Access detailed customer feedback visualizations through an external Tableau dashboard.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/MahmoudSaad21/SentiCall.git
cd arabic-sentiment-chatbot
```

### Step 2: Install Dependencies
Use the provided `requirements.txt` file to install all necessary libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment
- Place the pre-trained models (`vectorizer.joblib`, `logistic_model.joblib`, `bert_classifier.pth`) in the project directory.
- Add the dataset files (`CompanyReviews.csv`, `cleaned_customer_calls.csv`) to the project root.

### Step 4: Run the Application
Launch the Streamlit app using:
```bash
streamlit run app.py
```

---

## Usage

1. **Login Page:**
   - Use valid credentials (e.g., "Username: Mohamed Ammar, Password: 0000") to log in.
   
2. **Sentiment Analysis:**
   - Upload an Arabic audio file for transcription and sentiment analysis.
   - Select between BERT and MARBERT models for analysis.
   
3. **Customer Dataset Chatbot:**
   - Interact with a chatbot trained on customer data.
   - Ask questions and get insights from the data.

4. **Data Visualization:**
   - Explore customer feedback using dynamic charts and company-specific feedback trends.

5. **Tableau Dashboard:**
   - Access the external Tableau dashboard for additional visualizations.

---

## Technologies Used

- **Natural Language Processing (NLP):** BERT and MARBERT for sentiment analysis.
- **Speech Recognition:** Transcribe Arabic audio to text.
- **Google Generative AI:** Enable chatbot interactions.
- **Streamlit:** Interactive web application framework.
- **Data Visualization:** Plotly, Seaborn, and Tableau for charting and dashboards.

---

## Contributors

- **Mahmoud Saad Mahmoud** - Artificial Intelligence Engineer  
- **Mohamed Ahmed Ammar** - Machine Learning Engineer  
- **Shrouk Adel Mahmoud Mohamed** - Junior Machine Learning Engineer  
- **Fatma Saeed Foaad** - Machine Learning Engineer  
