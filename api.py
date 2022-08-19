# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import json
import codecs
import pandas as pd

import pickle
app = Flask(__name__)
# Load the model
modelCATB= pickle.load(open('WebApp\Final_Cat3.pkl' ,'rb'))
modelXGB = pickle.load(open('WebApp\Final_XGB_Improved_LF.sav','rb'))
modelRF = pickle.load(open('WebApp\Final_RF_LF2.pkl','rb'))
modelSVR = pickle.load(open('WebApp\Stack_XGB_CAT_RF.pkl','rb'))

def num(s):
    try:
        return int(s)
    except ValueError:
        return 0

@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.

    #data = request.json)
    #data = pd.DataFrame(data
    #data = pd.DataFrame(json.loads(codecs.decode(bytes(request.json.text, 'utf-8'), 'utf-8-sig'))['data'])
    #data = pd.read_csv(request.files.get('file'))
    #data = np.genfromtxt(request.files.get('file'), delimiter=',')
    #prediction = " CATBoost: " + np.array_str(modelCATB.predict(data)) + " XGBoost: " + "np.array_str(modelXGB.predict(data))" + "Random Forest: " + np.array_str(modelRF.predict(data))   # Take the first value of prediction

    result = dict(request.form)
    col = ['host_listings_count', 'bedrooms', 'accommodates', 'daysToLastReview'
     ,'calculated_host_listings_count_entire_homes', 'guests_included',
     'zipcode', 'bathrooms', 'number_of_reviews_ltm', 'calendar_updated2',
     'cleaning_fee', 'extra_people', 'harvesine_distance', 'host_id',
     'host_response_rate', 'calculated_host_listings_count', 'Elevator',
     'beds', 'daysToHostSince', 'security_deposit', 'Hangers',
     'Paid parking on premises', 'review_scores_rating', 'Shampoo',
     'Hot water', 'Coffee maker', 'Dishwasher', 'room_type_Entire home/apt',
     'room_type_Private room', 'room_type_Shared room',
     'neighbourhood_group_cleansed_Ballard',
     'neighbourhood_group_cleansed_Beacon Hill',
     'neighbourhood_group_cleansed_Capitol Hill',
     'neighbourhood_group_cleansed_Cascade',
     'neighbourhood_group_cleansed_Central Area',
     'neighbourhood_group_cleansed_Delridge',
     'neighbourhood_group_cleansed_Downtown',
     'neighbourhood_group_cleansed_Interbay',
     'neighbourhood_group_cleansed_Lake City',
     'neighbourhood_group_cleansed_Magnolia',
     'neighbourhood_group_cleansed_Northgate',
     'neighbourhood_group_cleansed_Other neighborhoods',
     'neighbourhood_group_cleansed_Queen Anne',
     'neighbourhood_group_cleansed_Rainier Valley',
     'neighbourhood_group_cleansed_Seward Park',
     'neighbourhood_group_cleansed_University District',
     'neighbourhood_group_cleansed_West Seattle',
     'host_response_time_a few days or more',
     'host_response_time_within a day',
     'host_response_time_within a few hours',
     'host_response_time_within an hour', 'host_location_California',
     'host_location_Others', 'host_location_Washington-Others',
     'host_location_Washington-Seattle', 'property_type_Aparthotel',
     'property_type_Apartment', 'property_type_Bed and breakfast',
     'property_type_Boat', 'property_type_Bungalow', 'property_type_Cabin',
     'property_type_Camper/RV', 'property_type_Condominium',
     'property_type_Cottage', 'property_type_Guest suite',
     'property_type_Guesthouse', 'property_type_Hostel',
     'property_type_House', 'property_type_Houseboat', 'property_type_Loft',
     'property_type_Other', 'property_type_Serviced apartment',
     'property_type_Tent', 'property_type_Tiny house',
     'property_type_Townhouse', 'cancellation_policy_strict',
     'cancellation_policy_strict_14_with_grace_period']


    data = np.array([num(result['slct_hostcount']), num(result['slct_bedroom']), num(result['slct_accomodates']), 0,
     num(result['slct_hostcountentirehome']), num(result['guests']),
     num(result['TXTBX_zip']), num(result['slct_bathrooms']), 16, 0, num(result['TXTBX_cleaningfee']), num(result['TXTBX_extra']), 12356, 0,
     100, num(result['slct_hostcount']),
     int(result.get('CHKBX_elevator') != None), num(result['slct_beds']), result['TXTBX_daysToHostSince'], result['TXTBX_secdepo'],
     int(result.get('CHKBX_hangers') != None),
     int(result.get('CHKBX_parking') != None), 10, int(result.get('CHKBX_shampoo') != None), int(result.get('CHKBX_water') != None), int(result.get('CHKBX_coffee') == None),
     int(result.get('CHKBX_Dishwasher') != None), int(result['slct_roomtype'] == 'Entire home/apt')
     , int(result['slct_roomtype'] == 'Private room'), int(result['slct_roomtype'] == 'Shared room'),
     int(result['slct_neighborhood'] == 'Ballard'),
     int(result['slct_neighborhood'] == 'Beacon Hill'), int(result['slct_neighborhood'] == 'Capitol Hill'),
     int(result['slct_neighborhood'] == 'Cascade'), int(result['slct_neighborhood'] == 'Central Area')
     , int(result['slct_neighborhood'] == 'Delridge'), int(result['slct_neighborhood'] == 'Downtown'),
     int(result['slct_neighborhood'] == 'Interbay'), int(result['slct_neighborhood'] == 'Lake City'),
     int(result['slct_neighborhood'] == 'Magnolia'), int(result['slct_neighborhood'] == 'Northgate'),
     int(result['slct_neighborhood'] == 'Other neighborhoods'), int(result['slct_neighborhood'] == 'Queen Anne'),
     int(result['slct_neighborhood'] == 'Rainier Valley'), int(result['slct_neighborhood'] == 'Seward Park'),
     int(result['slct_neighborhood'] == 'University District'), int(result['slct_neighborhood'] == 'West Seattle'),
     int(result['slct_hostresponsetime'] == 'a few days or more'), int(result['slct_hostresponsetime'] == 'within a day'),
     int(result['slct_hostresponsetime'] == 'within a few hours'),
     int(result['slct_hostresponsetime'] == 'within an hour'), int(result['slct_hostlocatione'] == 'California'),
     int(result['slct_hostlocatione'] == 'Others'),
     int(result['slct_hostlocatione'] == 'Washington-Others'), int(result['slct_hostlocatione'] == 'Washington-Seattle'),
     int(result['slct_propertytype'] == 'Aparthotel'),
     int(result['slct_propertytype'] == 'Apartment'), int(result['slct_propertytype'] == 'Bed and breakfast'),
     int(result['slct_propertytype'] == 'Boat'),
     int(result['slct_propertytype'] == 'Bungalow'), int(result['slct_propertytype'] == 'Cabin'),
     int(result['slct_propertytype'] == 'Camper/RV'),
     int(result['slct_propertytype'] == 'Condominium'), int(result['slct_propertytype'] == 'Cottage'),
     int(result['slct_propertytype'] == 'Guest suite'),
     int(result['slct_propertytype'] == 'Guesthouse'), int(result['slct_propertytype'] == 'Hostel'),
     int(result['slct_propertytype'] == 'House'), int(result['slct_propertytype'] == 'Houseboat'),
     int(result['slct_propertytype'] == 'Loft'), int(result['slct_propertytype'] == 'Other'),
     int(result['slct_propertytype'] == 'Serviced apartment'), int(result['slct_propertytype'] == 'Tent'),
     int(result['slct_propertytype'] == 'Tiny house'), int(result['slct_propertytype'] == 'Townhouse'),
     int(result['slct_cancellationpolicy'] == 'strict'), int(result['slct_cancellationpolicy'] == 'strict_14_with_grace_period')])
    data = data.astype(int)
    data = data.reshape(1, -1)
    modelRF.predict(data)
    preds_test = pd.DataFrame(
          {
              'predXGB': modelXGB.predict(pd.DataFrame(columns=col, data=data)),
              'predRF': modelRF.predict(data),
              'predCAT': modelCATB.predict(data)
          })
    prediction = np.round(modelSVR.predict(preds_test))[0].astype(int) # Take the first value of prediction
    return render_template('index.html', label=prediction)

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)