import numpy as np
import pandas as pd
import ipywidgets as widgets
#fit model with data
import matplotlib.pyplot as plt
import sklearn.linear_model

#negative/surprise: green space, air pollution, Household income, Life expectancy at birth
#yfactor = "Perceived health"

#stable: Households with internet access at home, Negative affect balance
#yfactor = "Feeling safe at night"  # not as much as I want
#yfactor = "Poor households without access to basic sanitary facilities"
yfactor = "Employment rate"
#yfactor = "Satisfaction with personal relationships"


#print("** all data **")
hls_all_raw = pd.read_csv("HSL.csv")
#print(hls_all_raw)

#print("** Only Indicator **")
#print(hls_all_raw["Indicator"])
#print("\n===========================================================\n")
#print("** slice **")
hls_slice = pd.DataFrame(hls_all_raw, columns =["Country","Indicator","Type of indicator","Time","Value"])
#print(hls_slice)

#print("** life satisfaction **")
#hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == "Life satisfaction"]
hls_ls = hls_slice.loc[hls_all_raw["Indicator"] == yfactor]

'''
print(hls_ls)
print("\n===========================================================\n")
print("Total records:")
print(len(hls_ls))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_ls["Country"].unique()))

print("\n===========================================================\n")
print("Country List")
print(hls_ls["Country"].unique())
'''

print("TRAINING")
hls_train = hls_ls.loc[hls_ls["Time"] == 2018]
#hls_train = hls_train.loc[hls_ls["Type of indicator"] == "Average"]
hls_train = hls_train.loc[hls_ls["Type of indicator"] == "Deprivation"]

print("\n===========================================================\n")
print("Total records:")
print(len(hls_train))

print("\n===========================================================\n")
print("Total Unique Countries:")
print(len(hls_train["Country"].unique()))

print("\n===========================================================\n")
print("Record:")
print(hls_train)

print("** LOADING GDP **")
weo_raw = pd.read_csv("GDP.csv")
#print(weo_raw)

print("** Subject Code **")
weo_selected_measurement = weo_raw.loc[weo_raw['WEO Subject Code'] == "NGDP_RPCH"]
weo_selected_measurement_2018 = pd.DataFrame(weo_selected_measurement, columns=['Country', '2018'])
print(weo_selected_measurement_2018)

print("** cleaning data")
merged_train_data = pd.merge(hls_train, weo_selected_measurement_2018, on="Country")
merged_train_data = merged_train_data.rename(columns={"Value": yfactor, "2018": "Income Measurement"})
merged_train_data = pd.DataFrame(merged_train_data, columns=['Country', yfactor, 'Income Measurement'])
print(merged_train_data)

print("** starting to plot")
X = np.c_[merged_train_data["Income Measurement"]]
Y = np.c_[merged_train_data[yfactor]]
x = X.tolist()
y = Y.tolist()

# plot data
out1 = widgets.Output()
with out1:
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel(yfactor)
  plt.title("Data Plot")
  plt.show()

# fit linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, Y)

# plot predictions
predict_x = [x for x in range(901)]
predict_x = [[x/100] for x in predict_x]
predict_y = model.predict(predict_x)

out2 = widgets.Output()
with out2:
  plt.scatter(predict_x, predict_y)
  plt.scatter(x, y)
  plt.xlabel('Income')
  plt.ylabel(yfactor)
  plt.title("Prediction Line")
  plt.show()

display(widgets.HBox([out1,out2]))