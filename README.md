# Premier-League-Win-Predictor
Made with Python, Pandas and Scikit Learn. Uses Web Scraping and Machine Learning concepts

Steps Taken To Develop The Final Product

1. Initial Web Scraping for a Single Team
2. Upscaled Web Scraping to get information for Every Team + Writing to a .csv file
3. Preliminary Steps for Machine Learning (reading .csv file, investigating missing data, cleaning data, creating predictors)
4. Training Initial Machine Learning Model (splitting training and test data sets, trained model using a Random Forest Algorithm on the dataset, analysed accuracy/precision data)
5. Improved Precision/Accuracy of Model for a Single Team (using rolling averages and creating new predictors with shooting stats)
6. Improved Precision/Accuracy of Model for Every Team
7. Retrained Machine Learning with rolling averages and new predictors
8. Combining Home & Away Predictions (to avoid redundancy within dataset and also eliminate possibilty of predictions being different for same match for the different teams)
9. Displaying Final Results of the Model
