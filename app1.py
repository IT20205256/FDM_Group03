import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv("C:\\Users\\Kishan raj\\Desktop\\Kishan\\fraud_oracle.csv")

data.loc[data["DayOfWeekClaimed"] == "0", "DayOfWeekClaimed"] = "NaN"
data.loc[data["MonthClaimed"] == "0", "MonthClaimed"] = "NaN"
data.loc[data["Age"] == 0, "Age"] = "NaN"

data.drop(data.index[data['DayOfWeekClaimed'] == "NaN"], inplace=True)
data.drop(data.index[data['MonthClaimed'] == "NaN"], inplace=True)
data.drop(data.index[data['Age'] == "NaN"], inplace=True)

#Function to convert Marital status into Numerical
def convertMarital(str):
 if str=="Single":
    return 0
 elif str=="Married":
    return 1
 elif str=="Divorced":
    return 2
 else:
       return 3
data["MaritalStatus"]= data["MaritalStatus"].apply(convertMarital)

#Function to convert Policy Type into Numerical
def convertPolicyType(str):
 if str=="Sedan - All Perils":
    return 0
 elif str=="Sedan - Collision":
    return 1
 elif str=="Sedan - Liability":
    return 2
 elif str=="Sport - All Perils":
    return 3
 elif str=="Sport - Collision":
    return 4
 elif str=="Sport - Liability":
    return 5
 elif str=="Utility - All Perils":
    return 6
 elif str=="Utility - Collision":
    return 7
 else:
       return 8
data["PolicyType"]= data["PolicyType"].apply(convertPolicyType)

#Function to convert Vehicle Category into Numerical
def convertVehicleCat(str):
 if str=="Sedan":
    return 1
 elif str=="Sport":
    return 2
 else:
       return 3
data["VehicleCategory"]= data["VehicleCategory"].apply(convertVehicleCat)

#Function to convert Vehicle Price into Numerical
def convertVehiclePrice(str):
 if str=="20000 to 29000":
    return 0
 elif str=="30000 to 39000":
    return 1
 elif str=="40000 to 59000":
    return 2
 elif str=="60000 to 69000":
    return 3
 elif str=="more than 69000":
    return 4
 else:
       return 5
data["VehiclePrice"]= data["VehiclePrice"].apply(convertVehiclePrice)

#Function to convert Past Claims into Numerical
def convertPastClaims(str):
 if str=="1":
    return 1
 elif str=="2 to 4":
    return 2
 elif str=="more than 4":
    return 3
 else:
       return 4
data["PastNumberOfClaims"]= data["PastNumberOfClaims"].apply(convertPastClaims)

#Function to convert Age of Policy Holder into Numerical
def convertPolicyHolderAge(str):
if str=="16 to 17":
    return 1
elif str=="18 to 20":
    return 2
elif str=="21 to 25":
    return 3
elif str=="26 to 30":
    return 4
elif str=="31 to 35":
    return 5
elif str=="36 to 40":
    return 6
elif str=="41 to 50":
    return 7
elif str=="51 to 65":
    return 8
else:
       return 9
data["AgeOfPolicyHolder"]= data["AgeOfPolicyHolder"].apply(convertPolicyHolderAge)

#Function to convert Police Report Filed into Numerical
def convertReport(str):
 if str=="No":
    return 0
 else:
       return 1
data["PoliceReportFiled"]= data["PoliceReportFiled"].apply(convertReport)

#Function to convert Witness Present into Numerical
def convertWitness(str):
 if str=="No":
    return 0
 else:
       return 1
data["WitnessPresent"]= data["WitnessPresent"].apply(convertWitness)

#Function to convert Agent Type into Numerical
def convertAgentType(str):
 if str=="External":
    return 0
 else:
       return 1
data["AgentType"]= data["AgentType"].apply(convertAgentType)

#Function to convert Address Change into Numerical
def convertAddressChange(str):
 if str=="no change":
    return 0
 elif str=="1 year":
    return 1
 elif str=="2 to 3 years":
    return 2
 elif str=="4 to 8 years":
    return 3
 else:
       return 4
data["AddressChange_Claim"]= data["AddressChange_Claim"].apply(convertAddressChange)

#Function to convert Base Policy into Numerical
def convertBasePolicy(str):
 if str=="All Perils":
    return 0
 elif str=="Collision":
    return 1
 else:
    return 2
data["BasePolicy"]= data["BasePolicy"].apply(convertBasePolicy)

#Function to convert Fault into Numerical
def convertFault(str):
 if str=="Policy Holder":
    return 0
 else:
       return 1
data["Fault"]= data["Fault"].apply(convertFault)

#Function to convert Days Policy Acc into Numerical
def convertPolicyAcc(str):
 if str=="1 to 7":
    return 0
 elif str=="8 to 15":
    return 1
 elif str=="16 to 30":
    return 2
 elif str=="more than 30":
    return 3
 else:
       return 4
data["Days_Policy_Accident"]= data["Days_Policy_Accident"].apply(convertPolicyAcc)

#Function to convert Days Policy Claim into Numerical
def convertPolicyClaim(str):
 if str=="8 to 15":
    return 0
 elif str=="15 to 30":
    return 1
 elif str=="more than 30":
    return 2
 else:
       return 3
data["Days_Policy_Claim"]= data["Days_Policy_Claim"].apply(convertPolicyClaim)

features = ["MaritalStatus","Fault", "PolicyType", "VehicleCategory", "VehiclePrice", "DriverRating", "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfPolicyHolder", "PoliceReportFiled",
         "WitnessPresent", "AgentType",  "AddressChange_Claim", "BasePolicy"]
x = data.loc[:,features]
y = data.loc[:,"FraudFound_P"]


# Using Random Forest to train the model
X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.75)


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
ranclf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
ranclf.fit(X_train1,y_train1)

y_pred1=ranclf.predict(X_test1)



import pickle
#Saving model to disk
pickle.dump(clf, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl','rb'))
