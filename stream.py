import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
import rarfile

os.system("apt-get install -y unrar")

def extract_rar_if_needed(rar_path, target_file):
    if not os.path.exists(target_file) and os.path.exists(rar_path):
        rf = rarfile.RarFile(rar_path)
        rf.extractall(".")

extract_rar_if_needed("zomato_classifier(3).rar", "zomato_classifier(3).pkl")

@st.cache_data
def load_model():
    return joblib.load(r"zomato_classifier(3).pkl")

@st.cache_data
def load_data():
    return pd.read_csv(r"zomato_clean(3).csv")

model = load_model()
df = load_data()

page = st.sidebar.radio("Choose page", ["Prediction", "EDA"])

if page == "Prediction":

    st.title("Zomato Restaurant Rate Classifier")
    st.write("Resturant Rating App")

    if st.checkbox("Display Data"):
        st.dataframe(df.head(20))

    st.markdown("---")

    st.subheader("Enter new resturant data for classification")

    categorical_cols = df.drop(columns=["rate", "rate_category"]).select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.drop(columns=["rate", "rate_category"]).select_dtypes(exclude=["object"]).columns.tolist()

    input_data = {}
    for col in categorical_cols:
        input_data[col] = st.selectbox(col, df[col].unique())

    for col in numerical_cols:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Predicted rating"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"predicted rating : *{prediction}*")

elif page == "EDA":
    st.subheader("EDA Page")
    st.header("1. Top 10 cuisines")
    image = Image.open(r"Top10Cuisines.png")
    st.image(image, width=700)

    st.header("3. Number of resturants per city")
    image = Image.open(r"RestaurantsperCity.png")
    st.image(image, width=700)

    st.header("4. Rating distribution per city")
    image = Image.open(r"RatingbyCity.png")
    st.image(image, width=700)

    st.header("5. the ratio of resturants that provide and dont provide table booking")
    image = Image.open(r"rationOfTableBooking.png")
    st.image(image, width=700)

    st.header("6. Rating distribution")
    image = Image.open(r"ratingDistribution.png")
    st.image(image, width=700)

    st.header("7. Online order availability")
    image = Image.open(r"1_online_order_pie.png")
    st.image(image, width=700)

    st.header("8. Restaurants per price band")
    image = Image.open(r"05_price_bands.png")
    st.image(image, width=700)

