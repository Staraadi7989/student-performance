# 🎓 EduPredict AI

A student performance prediction app built with Python, scikit-learn and Streamlit.

## Run in VS Code

1. Open this folder in VS Code (`File > Open Folder`).
2. Open the terminal (`Ctrl+` ` ) and create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Start the app:
   ```
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501. (Or use `Ctrl+Click` on the link in the terminal;
   VS Code's Simple Browser also works: `Ctrl+Shift+P` > "Simple Browser: Show".)

## Features
- Predicted score, grade, likely range, pass probability and class percentile
- Profile vs. top-performer radar chart and personalised recommendations
- What-If Lab: scenario sliders, best-single-change ranking, study-hours curve
- Insights: feature importance, score distribution, model accuracy, correlations
- Batch predictions from CSV (template included) with downloadable results
- Downloadable text report per student

## Use your own data
In `app.py`, replace `generate_data()` with `pd.read_csv("your_data.csv")`. Keep the same
column names (see the CSV template inside the app) plus a `final_score` column.
