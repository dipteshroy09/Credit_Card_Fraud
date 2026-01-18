# Credit Card Fraud Detection

A machine learning application for detecting credit card fraud using logistic regression with Streamlit interface.

## Features

- **Fraud Detection Model**: Trained logistic regression model with balanced class weights
- **Interactive Web Interface**: User-friendly Streamlit app for predictions
- **Flexible Input Methods**: Upload CSV files or use bundled dataset sample
- **Model Metrics**: View precision, recall, F1-score, and PR-AUC metrics
- **Adjustable Threshold**: Tune the decision threshold for fraud detection
- **Top Risky Transactions**: View the most suspicious transactions ranked by probability

## Project Structure

```
credit_card_fraud/
├── app/
│   └── app.py              # Streamlit application
├── src/
│   ├── train.py            # Model training script
│   └── inference.py        # Model loading and prediction functions
├── models/
│   ├── fraud_pipeline.joblib    # Trained model
│   └── fraud_metadata.json      # Model metadata
├── data/
│   └── creditcard.csv      # Dataset (not in repo if large)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd credit_card_fraud
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

Train the model using the provided script:

```bash
python src/train.py
```

The script will:
- Load data from `data/creditcard.csv`
- Split data into train/validation/test sets
- Train a logistic regression model with balanced class weights
- Find optimal threshold using F1-score on validation set
- Save the model to `models/fraud_pipeline.joblib`
- Save metadata to `models/fraud_metadata.json`

You can specify custom paths:
```bash
python src/train.py --data path/to/data.csv --model-out path/to/model.joblib --meta-out path/to/metadata.json
```

## Running Locally

Start the Streamlit app:

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Deployment

### Deploy to Streamlit Cloud

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and sign in with your GitHub account

3. **Click "New app"** and select:
   - Repository: Your repository
   - Branch: `main` (or your default branch)
   - Main file path: `app/app.py`
   - Python version: 3.11 or 3.12

4. **Click "Deploy"** - Streamlit Cloud will automatically:
   - Install dependencies from `requirements.txt`
   - Run your app
   - Provide a public URL

### Important Notes for Deployment

- **Model files**: The `models/` directory must be included in your repository. The model files (`.joblib` and `.json`) are small and should be committed to git.
- **Data file**: The `data/creditcard.csv` file is not included in the repo (per `.gitignore`). If you want to use the bundled dataset option, you'll need to either:
  - Include a sample of the data file in the repo
  - Or remove the "Use bundled dataset sample" option from the app
- **Secrets**: If you need environment variables or secrets, configure them in Streamlit Cloud's settings

### Alternative Deployment Options

- **Heroku**: Use a `Procfile` with `web: streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0`
- **Docker**: Create a Dockerfile with Streamlit installed
- **AWS/GCP/Azure**: Deploy as a containerized application

## Model Details

- **Algorithm**: Logistic Regression with balanced class weights
- **Preprocessing**: StandardScaler for feature normalization
- **Threshold Selection**: Optimized for F1-score on validation set
- **Features**: Time, V1-V28 (PCA components), Amount
- **Target**: Class (0 = Normal, 1 = Fraud)

## Requirements

See `requirements.txt` for full dependency list. Key packages:
- streamlit >= 1.42.0
- scikit-learn >= 1.8.0
- pandas >= 2.3.3
- numpy >= 2.4.0
- joblib >= 1.5.3
- pillow >= 11.0.0, < 12

## License

[Your License Here]

## Contact

[Your Contact Information]
