# 💼 IBM Employee Salary Prediction -- AICTE

An AI-powered web application that predicts employee salaries based on job role, education, experience, and gender. This project was developed under the **AICTE–IBM Internship Program** using machine learning, and offers a simple and intuitive interface to analyze and visualize salary trends.

![Salary Prediction]
(https://github.com/sanjiv-0103/EmployeeSalaryPrediction---B-Sanjiv.git)

---

## Features ✨ 

- 🎯 Predict Monthly and Annual Salary based on user inputs
- 📊 Interactive Dashboard with individual and CSV-based predictions
- 📈 Salary Growth Graph based on years of experience
- 📁 Upload Your Dataset and get salary predictions in bulk
- 🧠 Smart Label Encoding with pre-trained encoders for consistent results
- 💼 Realistic Range Limiting to prevent extreme prediction outputs
- 📥 Downloadable Reports in TXT or CSV format for offline access
- 🌐 Responsive UI powered by Streamlit and Plotly
- 🧪 Random Forest Model trained on demographic and professional attributes



---

## 🛠️ Tech Stack

| Tool              | Purpose                            |
|-------------------|------------------------------------|
| Python 3.11+       | Core programming language         |
| Streamlit         | Web interface                      |
| scikit-learn      | Machine Learning model             |
| pandas & NumPy    | Data manipulation                  |
| plotly            | Interactive visualizations         |
| joblib            | Model serialization                |
| Git + GitHub      | Version control & collaboration    |

---

## 📂 Project Structure

AICTE_IBM_Salary_Prediction/
├── EmployeeSalaryPrediction.py    # Streamlit app
├── best_model.pkl    # ML model
├── trained_columns.pkl   # List of columns used for training
├── label_encoders.pkl   # Label encoders for categorical features
├── EmployeeSalaryPrediction(2).ipynb # Model building jupyter notebook
├── README.md # Project documentation
└── assets/ 

---

## 📈 Visualizations

| Chart Type        | Description                                         |
|-------------------|-----------------------------------------------------|
| 📈 Line Chart     | Shows salary progression over 0–20 years experience|
| 📂 Custom Dataset | Analyzes trends from uploaded files                |


🔧 Project Setup Instructions

✅ 1. Pre-requisites
Make sure you have the following installed on your system:

Python 3.10
Github
pip (Python package manager)
virtualenv -- virtual environment (optional)


📥 2. Clone the Repository
Open your terminal or command prompt and run:

git clone (https://github.com/sanjiv-0103/EmployeeSalaryPrediction---B-Sanjiv.git)
cd EmployeeSalaryPrediction


📦 3. Set Up a Virtual Environment (Recommended)

### ➡ Create virtual environment
python -m venv venv

### ➡ Activate the virtual environment
venv\Scripts\activate  (For Windows)

📚 4. Install Dependencies
Make sure to install required packages. If requirements.txt is not present, install manually:

pip install streamlit scikit-learn pandas numpy plotly joblib
Or if requirements.txt exists:

pip install -r requirements.txt


🚀 5. Run the Streamlit App

streamlit run app.py
After a few seconds, the app will open in your default web browser at:
http://localhost:8501

