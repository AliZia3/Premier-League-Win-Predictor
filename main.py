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
    years = list(range(2023, from_year, -1))
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


def MakePredictions(matches, predictors):
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

    train = matches[matches["Date"] < "2023-01-01"]
    test = matches[matches["Date"] > "2023-01-01"]

    rf.fit(train[predictors], train["Target"])
    preds = rf.predict(test[predictors])

    combined = pd.DataFrame(dict(actual=test["Target"], predicted=preds), index=test.index).merge(matches[["Date", "Team", "Opponent", "Result"]], left_index=True, right_index=True)
    merged = PredictionDataCleaning(combined)
    merged.to_csv("results.csv")

    acc = accuracy_score(test["Target"], preds)
    precision = precision_score(test["Target"], preds)
    crosstab = pd.crosstab(index=combined["actual"], columns=combined["predicted"])

    return merged, acc, precision, crosstab



def MLModel(match_data_file):
    cleaned_match_data = DataCleaning(match_data_file)
    matches, predictors = MatchPredictors(cleaned_match_data)
    matches_rolling = RollingAveragesTeam(matches)

    return MakePredictions(matches_rolling, predictors + new_cols)




def Main():
    print("PREMIER LEAGUE WIN PREDICTOR")
    print("q: Quit \nscrape: Scrape Premier League Data (Recommended if matches.csv file isnt available might not be able to scrape many seasons due to scraping policy of website) \nresults: See Results (Downloads csv file of results as well)\n")
    match_data_file = 'matches.csv'

    command = input("Enter a command: ")


    while(True):
        while (command != 'q' and command != 'scrape' and command != 'results'):
            print("Invalid Command...")
            command = input("Enter a command: ")
        
        if (command == 'q'):
            print("Exiting...")
            exit()

        if (command == 'scrape'):    
            to_year = 2023
            print("All data will be to 2023 (Present Season)")
            from_year = int(input("Results you want from year: "))
            
            while(from_year > to_year):
                print("From year has to be less than To Year (2023)")
                from_year = input("Results you want from year: ")
            
            try:
                print("Getting Data (This may take a couple of minutes)...")
                WebScraper(from_year)
                print("Data Scraped\n")
                print("q: Quit \nscrape: Scrape Premier League Data (Recommended if matches.csv file isnt available might not be able to scrape many seasons due to scraping policy of website) \nresults: See Results (Downloads csv file of results as well)\n")
                command = input("Enter a command: ")
            except:
                print("Couldnt Scrape. Likely due to websites scraping policy. Try changing the to and from years to a smaller amount\n")
                print("q: Quit \nscrape: Scrape Premier League Data (Recommended if matches.csv file isnt available might not be able to scrape many seasons due to scraping policy of website) \nresults: See Results (Downloads csv file of results as well)\n")
                command = input("Enter a command: ")

        if(command == 'results'):
            file_found = False
            try:
                Merged, Acc, Precision, Crosstab = MLModel(match_data_file)
                file_found = True
            except:
                print("\n*(matches.csv file not found, consider scraping the data or downloading the csv file from my github repo)")
                print("q: Quit \nscrape: Scrape Premier League Data (Recommended if matches.csv file isnt available might not be able to scrape many seasons due to scraping policy of website) \nresults: See Results (Downloads csv file of results as well)\n")
                command = input("Enter a command: ")

            if(file_found):
                print("\nback: Go Back \nmerged: Check Final Merged Data, \naccuracy: Check Model Accuracy, \nprecision: Check Model Precision, \ncrosstab: Check Crosstab of Predictions\n")
                inner_command = input("Enter a command: ")
                
                while (inner_command != 'back' and inner_command != 'merged' and inner_command != 'accuracy' and inner_command != 'precision' and inner_command != 'crosstab'):
                    print("Invalid Command...")
                    inner_command = input("Enter a command: ")
                    
                
                if(inner_command == 'back'):
                    print("q: Quit \nscrape: Scrape Premier League Data (Recommended if matches.csv file isnt available might not be able to scrape many seasons due to scraping policy of website) \nresults: See Results (Downloads csv file of results as well)\n")
                    command = input("Enter a command: ")
                if (inner_command == 'merged'):
                    print(Merged)
                if (inner_command == 'accuracy'):
                    print(Acc)
                if (inner_command == 'precision'):
                    print(Precision)
                if (inner_command == 'crosstab'):
                    print(Crosstab)

Main()