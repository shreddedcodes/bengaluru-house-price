
# Bengaluru House Price Prediction

This model uses XGBoost with random search for hyperparameter tuning to predict the price of a house in Bengaluru, given the features.

The app is deployed on a platform called Streamlit which can either be run from the local host or using the cloud platform.

To run it locally

## Deployment

To deploy this project run

```bash
  conda create --name <env> --file ml_env.txt
  conda activate ml_env
  streamlit run app.py
```

Or you can access it via the cloud using the link below:
https://shreddedcodes-bengaluru-house-price-app-gmixff.streamlit.app/