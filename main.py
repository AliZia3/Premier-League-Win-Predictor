# Imports
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
# pip install above libraries
# Also pip install lxml to parse html

def WebScraper(from_year):
    years = list(range(2023, int(from_year), -1))
    # years = list(range(to_year, from_year, -1))

    all_matches = []

    standings_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'

    for year in years:
        data = requests.get(standings_url)
        bs = BeautifulSoup(data.text, features="lxml")
        
        standings_table = bs.select('table.stats_table')[0]
        links = [a.get('href') for a in standings_table.find_all('a')]
        links = [a for a in links if '/squads/' in a]
        team_urls = [f'https://fbref.com{a}' for a in links]
        
        previous_season = bs.select("a.prev")[0].get('href')
        standings_url = f'https://fbref.com{previous_season}'
        
        for team_url in team_urls:
            
            team_name = team_url.split('/')[-1].replace('-Stats', '').replace('-', ' ')
            
            data = requests.get(team_url)
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
            
            bs = BeautifulSoup(data.text, features="lxml")
            links = [a.get("href") for a in bs.find_all('a')]
            links = [a for a in links if a and 'all_comps/shooting/' in a]
            data = requests.get(f"https://fbref.com{links[0]}")
            shooting = pd.read_html(data.text, match='Shooting')[0]
            shooting.columns = shooting.columns.droplevel() 
                
            try:
                team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
            except ValueError:
                continue
                    
            team_data = team_data[team_data["Comp"] == "Premier League"]
            team_data["Season"] = year
            team_data["Team"] = team_name
                
            all_matches.append(team_data)
            time.sleep(5)

    match_df = pd.concat(all_matches)

    match_df.to_csv("matches.csv")



def DataCleaning(match_data_file):
    matches = pd.read_csv(match_data_file, index_col=0)
    matches["Date"] = pd.to_datetime(matches["Date"])
    del matches["Comp"]
    del matches["Notes"]
    del matches["Referee"]
    del matches["Match Report"]
    del matches["Captain"]
    del matches["Formation"]
    del matches["Attendance"]
    del matches["Poss"]

    return matches



def MatchPredictors(matches):
    matches["Venue_code"] = matches["Venue"].astype("category").cat.codes
    matches["Opp_code"] = matches["Opponent"].astype("category").cat.codes
    matches["Hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype('int')
    matches["Day_code"] = matches["Date"].dt.dayofweek
    matches["Target"] = (matches["Result"] == "W").astype("int")

    predictors = ["Venue_code", "Opp_code", "Hour", "Day_code"]

    return matches, predictors



def MatchStatsPredictors():
    cols = ["xG", "xGA", "GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
    global new_cols 
    new_cols = [f"{c}_rolling" for c in cols]

    return [cols, new_cols]



def RollingAverages(team):
    cols, new_cols = MatchStatsPredictors()[0], MatchStatsPredictors()[1]
    team = team.sort_values("Date")
    rolling_stats = team[cols].rolling(3, closed='left').mean()
    team[new_cols] = rolling_stats
    team = team.dropna(subset=new_cols)
    
    return team



def RollingAveragesTeam(matches):
    matches_rolling = matches.groupby("Team").apply(lambda team: RollingAverages(team))
    matches_rolling = matches_rolling.droplevel('Team')
    matches_rolling.index = range(matches_rolling.shape[0])

    return matches_rolling



class MissingDict(dict):
    __missing__ = lambda self, key: key



def PredictionDataCleaning(combined_prediction_results):
    map_values = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Newcastle United": "Newcastle Utd",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves"    
    }
    mapping = MissingDict(**map_values)
    combined_prediction_results["Team"] = combined_prediction_results["Team"].map(mapping)

    merged_prediction_results = combined_prediction_results.merge(combined_prediction_results, left_on=["Date", "Team"], right_on=["Date", "Opponent"])

    return merged_prediction_results



def MakePredictions(matches, predictors, predictionType):
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

    train = matches[matches["Date"] < "2023-01-01"]
    test = matches[matches["Date"] > "2023-01-01"]

    rf.fit(train[predictors], train["Target"])
    preds = rf.predict(test[predictors])

    if predictionType=="with_actuals":
        combined = pd.DataFrame(dict(actual=test["Target"], predicted=preds), index=test.index).merge(matches[["Date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)
        crosstab = pd.crosstab(index=combined["actual"], columns=combined["predicted"])
        acc = accuracy_score(test["Target"], preds)
        precision = precision_score(test["Target"], preds)
    else:
        combined = pd.DataFrame(dict(predicted=preds), index=test.index).merge(matches[["Date", "Team", "Opponent"]], left_index=True, right_index=True)
        crosstab = None
        acc = None
        precision = None

    merged = PredictionDataCleaning(combined)

    return merged, acc, precision, crosstab


def MLModel(match_data_file):
    cleaned_match_data = DataCleaning(match_data_file)
    matches, predictors = MatchPredictors(cleaned_match_data)
    matches_rolling = RollingAveragesTeam(matches)

    return MakePredictions(matches_rolling, predictors + new_cols, "with_actuals")


def CustomMLModel(prediction_data_file):
    predictors = ["Venue_code", "Opp_code", "Hour", "Day_code"]
    matches = pd.read_csv(prediction_data_file, index_col=0)
    matches.index = range(matches.shape[0])

    return MakePredictions(matches, predictors, "without_actuals")



def main():
    print("PREMIER LEAGUE WIN PREDICTOR")
    match_data_file = 'matches.csv'
    prediction_data_file = 'predict.csv'

    while(True):
        print("*SOME IMPORTANT INFORMATION: The ML model can serve 2 purposes -> \n")
        print("1. Making predictions for already existing data/data you scrape through the webscraper This means you have a matches.csv file (This can help you gain a better idea of the accuracy and precision of the model)")
        print("2. You can create your own predict.csv file (like the one currently provided) this can help you make predictions for future games")
        print("===============================================================================================================================================================================================================\n")
        print("q: Quit \nscrape: Scrape Premier League Data (Recommended if matches.csv file isnt available might not be able to scrape many seasons due to scraping policy of website) \nresults: See Results of matches.csv (Downloads csv file of results as well) \npredict: See results of predict.csv (downloads csv file of results as well)")
        command = input("Enter a command: ")
        
        if (command == 'q'):
            print("Exiting...")
            exit()

        elif (command == 'scrape'):    
            to_year = 2023
            print("All data will be to 2023 (Present Season)")
            from_year = input("Results you want from year: ")
            
            while(from_year > to_year):
                print("From year has to be less than To Year (2023)")
                from_year = input("Results you want from year: ")
            
            try:
                print("Getting Data (This may take a couple of minutes)...")
                WebScraper(from_year)
                print("Data Scraped\n")
            except:
                print("Couldnt Scrape. Likely due to websites scraping policy. Try changing the to and from years to a smaller amount\n")

        elif(command == 'results'):
            try:
                Merged, Acc, Precision, Crosstab = MLModel(match_data_file)
                Merged.to_csv("results.csv")

                while(True):
                    print("\nback: Go Back \nmerged: Check Final Merged Data, \naccuracy: Check Model Accuracy, \nprecision: Check Model Precision, \ncrosstab: Check Crosstab of Predictions\n")
                    inner_command = input("Enter a command: ")

                    if(inner_command == 'back'):
                        break
                    elif (inner_command == 'merged'):
                        print(Merged)
                    elif (inner_command == 'accuracy'):
                        print(Acc)
                    elif (inner_command == 'precision'):
                        print(Precision)
                    elif (inner_command == 'crosstab'):
                        print(Crosstab)
                    else:
                        print("Invalid Command...")

            except FileNotFoundError:
                print("\n*(matches.csv file not found, consider scraping the data or downloading the csv file from my github repo)")
        
        elif(command=="predict"):
            try:
                Merged, Acc, Precision, Crosstab = CustomMLModel(prediction_data_file)
                Merged.to_csv("predictions.csv")
                
                while(True):
                    print("\nback: Go Back \nmerged: Check Final Merged Data\n")
                    inner_command = input("Enter a command: ")

                    if(inner_command == 'back'):
                        break
                    elif (inner_command == 'merged'):
                        print(Merged)
                    else:
                        print("Invalid Command...")

            except FileNotFoundError:
                print("\n*(matches.csv or predict.csv file not found, consider downloading the matches.csv and predict.csv file from my github repo)")


        else:
            print("Invalid Command...")


if __name__ == "__main__":
    main()