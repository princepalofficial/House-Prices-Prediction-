import numpy as np
import pandas as pd
# 1. create house Dataset
np.random.seed(42)
n = 1000
data = {"house_id": np.arange(1, n + 1),
        "Area_sqft": np.random.randint(500,3501,n),
        "bedrooms": np.random.randint(1,6,n),
        "bathrooms": np.random.randint(1,4,n),
        "age": np.random.randint(0, 31,n),
        "Parking": np.random.randint(0,3,n),
        "location_score": np.random.randint(1,11,n)}

df = pd.DataFrame(data)
# 2. create house price
df["price"] = (df["Area_sqft"] * 300 + df["bedrooms"] * 300000 + df["bathrooms"] * 200000
               + df["Parking"] * 100000 + df["location_score"] * 250000 - df["age"] * 50000)

# Add randam market variation
noise = np.random.randint(-300000,300001, n)
df["price"] = df["price"] + noise

# positive price
df["price"] = np.maximum(df["price"],500000)

# 3. Basic data checking
print("\n ****** First 5 houses ****** ")
print(df.head())
print("\n ****** Dataset shape ****** ")
print(df.shape)
print("\n ******columns ******")
print(df.columns)
print("\n ****** Data type ******")
print(df.dtypes)
print("\n ******* missing values ******")
print(df.isnull().sum())
print("\n ****** Duplicated rows *******")
print(df.duplicated().sum())
print("\n ****** Statistics ******")
print(df.describe())

# 4. Average House price
average_price = df["price"].mean()
print("\n ******* Average price ******")
print(round(average_price, 2))

# 5. Feature Engineering
df["price_per_sqft"] = df["price"] / df["Area_sqft"]
df["large_house"] = np.where(df["Area_sqft"] >= 2000, "Yes","No")
df["new_house"] = np.where(df["age"] <= 5, "Yes", "No")
df["good_location"] = np.where(df["location_score"] >= 7, "Yes","No")

# 6. House price analysis
print("\n *********** price by bedrooms ************")
bedrooms_price = df.groupby("bedrooms")["price"].mean()
print(bedrooms_price.round(2))

print("\n ************ price by bathroom ************")
bathroom_price = df.groupby("bathrooms")["price"].mean()
print(bathroom_price.round(2))

print("\n ********* price by parking *********")
parking_price = df.groupby("Parking")["price"].mean()
print(parking_price.round(2))

# 7. Large house analysis
print("\n ******* Large House Analysis ********")
large_house_price = df.groupby("large_house")["price"].mean()
print(large_house_price.round(2))

# 8. Location Analysis
print("\n ******* Locations Analysis **********")
location_price = df.groupby("good_location")["price"].mean()
print(location_price.round(2))

# 9. Manual price prediction
df["predicted_price"] = (df["Area_sqft"] * 300 + df["bedrooms"] * 300000 + df["bathrooms"] * 200000 
                         + df["Parking"] * 100000 + df["location_score"] * 250000 - df["age"] * 50000 )

df["predicted_price"] = np.maximum(df["predicted_price"],500000 )

# 10. Actual vs predicted
df["difference"] = (df["price"] - df["predicted_price"])

print("\n ******** Actul vs predicted *******")
print(df[["house_id","Area_sqft","bedrooms","bathrooms","price","predicted_price","difference"]].head(10))

# 11. MAE calcuation
mae = np.mean(np.abs(df["price"] - df["predicted_price"]))
print("\n****** mae **********")
print(round(mae, 2))

# 12. New House price prediction
new_house = pd.DataFrame({"area_sqft":[1800],"bedrooms": [3],"bathrooms":[2],
                          "age":[5],"parking":[1],"location_score":[8]})
new_house["predicted_price"] = (new_house["area_sqft"] * 300 + new_house["bedrooms"] * 300000
+ new_house["bathrooms"] * 200000 + new_house["parking"] * 100000 + new_house["location_score"] * 250000
- new_house["age"] * 50000)

new_house["predicted_price"] = np.maximum(new_house["predicted_price"],500000)
print("\n ******* new house ******")
print(new_house)

print("\n ********** predicted house price ********")
price = new_house["predicted_price"].iloc[0]
print("Rupies", round(price, 2))