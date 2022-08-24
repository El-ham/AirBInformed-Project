# AirBInformed-Project
An app which accurately predicts Airbnb rental prices.
The data that I have worked on is form Insideairbnb website. I chose Seattle for which we have 16213 houses and 364 features.
These features can be divided into 3 groups. First, the features related to the house itself, for example number of bathrooms and its amenities. Second, the features related to its location, like its neighborhood and zip code. Third, the features related to the host like where the location of the host is or how responsive s/he is.
After cleaning and preparing the dataset, I separated 25% of the data for testing. I used three regression models in parallel using grid search and 5-fold cross validation on the remaining 75%.
Finally, stacking has been performed to combine the predictions obtain by the models.
The result of the models which are: Random Forest, CATBoost and XGBoost is shown in the slide show available in this repo. The adjusted R^2 from stacking is 0.91.
Because in a real world scenario, the user cannot enter many features to see their prediction, based on the result of the model, I have selected the most important features which provide more than 70% gain in comparison to each original models.
So, the number of features reduced to 28. After feature selection, running the models and stacking again, based on the fact that the average income from houses is about 3000$ per month, we have 14% error in our prediction.
There were many challenges in this project but data cleaning has been the most challenging part. There were a lot of features with NaN or only one value. A lot of features had high correlation with others.
There were a lot of values with extra characters which had to be fixed. Many dummy variables were created not only based on categorical features but also features which had a list for their value.
Aggregation was another challenge because I had multiple monthly snapshot of the listings so I had manually aggregate each feature for each property. 


