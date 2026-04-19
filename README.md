# Financial Distress AI Agent

An intelligent AI agent that predicts financial distress in companies by analyzing financial statements. The system allows users to upload Excel/CSV files containing financial data, provides risk predictions, offers visual analysis, generates summary reports, and includes a finance-specialized chatbot for answering questions.

## Features

- **Financial Statement Analysis**: Upload CSV or Excel files containing company financial data
- **AI-Powered Risk Prediction**: Uses a trained XGBoost model to predict financial distress
- **Credit Scoring System**: Converts predictions to intuitive credit scores (0-100)
- **Risk Categorization**: Automatic classification into LOW, MEDIUM, HIGH, and VERY HIGH risk levels
- **Interactive Visualizations**: Risk distribution charts, credit score histograms, feature importance plots
- **Explainability**: SHAP values for understanding model predictions at the company level
- **Finance Chatbot**: AI assistant that answers finance and risk-related questions
- **Report Generation**: Downloadable PDF reports with company risk summaries
- **Database Storage**: Prediction history stored in SQLite database
- **Filtering & Sorting**: Interactive filtering by risk levels

## Project Structure

```
financial-distress-ai/
│
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── model/                  # Trained model artifacts
│   ├── distress_model.pkl  # Trained XGBoost model
│   └── scaler.pkl          # Feature scaler
│
├── data/                   # Data files
│   └── american_bankruptcy.csv  # Training dataset
│
├── .venv/                  # Python virtual environment
├── .vscode/                # VS Code configuration
└── __pycache__/            # Python cache files
```

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd financial-distress-ai
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key** (for chatbot functionality):
   - Create a `.streamlit/secrets.toml` file in the project root
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "your-openai-api-key-here"
     ```
   - Alternatively, configure Streamlit secrets through your deployment platform

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload financial data**:
   - Use the file uploader to select a CSV or Excel file containing financial statements
   - The system expects financial ratios and metrics as columns (similar to the training data format)

3. **View results**:
   - See raw data preview
   - Filter results by risk level
   - View visualizations (risk distribution, credit score distribution, feature importance)
   - Analyze individual companies with SHAP explanations
   - Download PDF reports

4. **Use the chatbot**:
   - Ask finance-related questions in the sidebar
   - Get expert-like responses from the AI credit risk analyst

## Model Information

The prediction model is an XGBoost classifier trained on financial distress data. Key features:
- Handles class imbalance using SMOTE
- Includes outlier removal and data normalization
- Optimized threshold for better risk classification
- Provides probability outputs converted to credit scores

## Dependencies

See `requirements.txt` for the complete list:
- streamlit: Web application framework
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning utilities
- xgboost: Gradient boosting model
- imblearn: Handling imbalanced datasets (SMOTE)
- shap: Model explainability
- reportlab: PDF generation
- openpyxl: Excel file handling
- matplotlib, seaborn: Data visualization
- openai: Chatbot functionality

## How It Works

1. **Data Processing**:
   - Uploaded financial data is cleaned (missing values filled with median)
   - Features are scaled using the pre-trained scaler
   - Company names are removed if present (for privacy)

2. **Prediction**:
   - Scaled data is fed into the XGBoost model
   - Model outputs probability of financial distress
   - Probability converted to credit score: `score = (1 - probability) * 100`
   - Scores mapped to risk categories:
     - 80-100: LOW RISK 🟢
     - 60-79: MEDIUM RISK 🟡
     - 40-59: HIGH RISK 🟠
     - 0-39: VERY HIGH RISK 🔴

3. **Explainability**:
   - SHAP values show feature contributions to individual predictions
   - Helps understand which financial metrics drive risk assessments

4. **Chatbot**:
   - Uses OpenAI's GPT-4o-mini model
   - Specialized system prompt for credit risk analysis
   - Can answer questions about specific companies, risk factors, or general finance topics

## Customization

- **Model Retraining**: Run `train_model.py` to retrain with new data
- **Feature Engineering**: Modify preprocessing steps in `app.py` or `train_model.py`
- **Risk Thresholds**: Adjust risk band functions in `app.py`
- **Chatbot Prompt**: Edit the system message in the chatbot section of `app.py`

## Notes

- The model expects financial ratio data similar to the bankruptcy prediction dataset
- For best results, ensure uploaded data has similar feature distribution to training data
- SHAP visualizations may require additional dependencies in some environments
- PDF report generation shows first 10 companies by default

## Future Improvements

- Add more sophisticated financial statement parsing
- Include time-series analysis for tracking company health over time
- Implement industry-specific risk models
- Add batch processing capabilities
- Enhance PDF report formatting and customization options

---

**Built with Streamlit and XGBoost for intelligent financial risk analysis.**