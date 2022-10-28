import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'MaritalStatus:','Fault:', 'PolicyType:', 'VehicleCategory:', 'VehiclePrice:', 'DriverRating:', 'Days_Policy_Accident:', 'Days_Policy_Claim:', 'PastNumberOfClaims:', 'AgeOfPolicyHolder:', 'PoliceReportFiled:',
         'WitnessPresent:', 'AgentType:',  'AddressChange_Claim:', 'BasePolicy:'})

print(r.json())