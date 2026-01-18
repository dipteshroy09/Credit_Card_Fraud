# Deployment Guide

This guide provides detailed instructions for deploying the Credit Card Fraud Detection app to various platforms.

## Streamlit Cloud (Recommended - Free)

Streamlit Cloud is the easiest and fastest way to deploy Streamlit apps for free.

### Prerequisites
- GitHub account
- Code pushed to a GitHub repository

### Step-by-Step Instructions

1. **Prepare your repository**:
   - Ensure `requirements.txt` is in the root directory
   - Make sure model files (`models/fraud_pipeline.joblib` and `models/fraud_metadata.json`) are committed
   - Ensure `.streamlit/config.toml` exists (optional but recommended)

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

3. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with your GitHub account
   - Click "New app"
   - Fill in:
     - **Repository**: Select your repository
     - **Branch**: `main` (or your default branch)
     - **Main file path**: `app/app.py`
   - Click "Deploy"

4. **Access your app**:
   - Your app will be available at: `https://<your-app-name>.streamlit.app`
   - The app will automatically redeploy when you push changes to the connected branch

### Troubleshooting Streamlit Cloud

- **Missing model files**: Ensure `models/fraud_pipeline.joblib` and `models/fraud_metadata.json` are committed to git
- **Import errors**: Verify all dependencies are in `requirements.txt`
- **Data file issues**: The `data/creditcard.csv` file is ignored by git. Either:
  - Include a sample data file
  - Use only the "Upload CSV" option in the app
  - Add a smaller sample dataset to the repository

## Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t fraud-detection-app .
docker run -p 8501:8501 fraud-detection-app
```

## Heroku Deployment

1. **Create a `Procfile`**:
   ```
   web: streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create `runtime.txt`** (optional):
   ```
   python-3.11.0
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Manual Server Deployment

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with Streamlit**:
   ```bash
   streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
   ```

3. **Use a process manager** (PM2, systemd, etc.) to keep it running

## Environment Variables

If needed, you can configure these environment variables:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

## Security Considerations

- Don't commit sensitive data or API keys
- Use Streamlit Cloud's secrets management for credentials
- Enable authentication if deploying sensitive applications
- Consider rate limiting for public deployments

## Performance Optimization

- Model files are cached using `@st.cache_resource`
- Data loading is cached using `@st.cache_data`
- Consider using a CDN for static assets
- Use Streamlit Cloud's resources for production workloads
