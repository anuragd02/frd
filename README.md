# Arecanut Fruit Rot Predictor (Streamlit app)

Estimates the same-week **CFNR** (fallen nuts per tree) and **disease severity (%)** of arecanut fruit rot. Inputs are the region, the week of the season (W1–W18) and one week of weather. Last week's CFNR and severity can optionally be added.

The model is Extra Trees, trained on 2018–2019 data from 27 locations in the Malnad, Coastal and Maidan regions.

Run locally:

```
pip install -r requirements.txt
streamlit run app.py
```
