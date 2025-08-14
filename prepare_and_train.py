import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.neural_network import MLPClassifier

df = pd.read_csv(r"zomato.csv")

df.drop(columns=['url', 'address', 'name', 'phone', 'reviews_list', 'menu_item', 'dish_liked'], inplace= True, errors= "ignore")


df.columns = df.columns.str.strip()


df.dropna(how="all", inplace=True)


df['approx_cost(for two people)'] = (df['approx_cost(for two people)'].astype(str).str.replace(',', '').astype(float))

df['rate'] = (df['rate'].astype(str).replace(['NEW', '-'], np.nan).str.replace('/5', '').astype(float))

df.dropna(subset=['location'], inplace=True)
df.dropna(subset=['rest_type'], inplace=True)
df.dropna(subset=['cuisines'], inplace=True)
df.dropna(subset=['approx_cost(for two people)'], inplace=True)

df.fillna("Unknown", inplace=True)

df.drop_duplicates(inplace=True)



def clean_rate(x):
    try:
        val = str(x).split('/')[0]   
        return float(val)
    except:
        return np.nan

df["rate"] = df["rate"].apply(clean_rate)


df['rate'].fillna(df['rate'].median(), inplace=True)

def rate_category(r):
    if r < 3.5:
        return "Low"
    elif r < 4.0:
        return "Medium"
    else:
        return "High"



df["rate_category"] = df["rate"].apply(rate_category)

target = "rate_category"
X = df.drop(columns=[target, "rate"])
y = df[target]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

nn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), activation = 'relu', solver = 'adam', max_iter = 300, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "zomato_classifier(4).pkl")
df.to_csv("zomato_clean(4).csv", index=False)

print("\n file saved 'zomato_classifier.pkl'")
print("file saved 'zomato_clean.csv'")


nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
print("NN_Accuracy:", accuracy_score(y_test, nn_pred))
print("\nNN_Classification Report:\n", classification_report(y_test, nn_pred))

joblib.dump(nn_model, "NN_zomato_classifier(4).pkl")
print("\n file saved 'NN_zomato_classifier.pkl'")
