# Data Science

Analysis of Cricket World Cup Games & Predict the winner in each match.   
This project is based on _result.csv, icc_rankings.csv, fixtures.csv_

# Dataset Description 
This data set consist of ICC Cricket world cup 2019 stats like matches, teams. past matches, data for the winning teams from 2010.    This data set is used in logistic regression algorithm to predict the winning team of 2019 Cricket World Cup played in England.   
   
Dataset reference : https://www.kaggle.com/saadhaxxan/cwc-world-cup-2019-prediction-data-set

# Data preprocessing
The data was dropped and scaled (StandardScaler, MinMaxScaler, RobustScaler) to go through data preprocessing.
~~~
results.dropna(axis = 0, inplace = True)
results.drop(['Ground'], axis = 1, inplace = True)
~~~
It created and used an automation function that scales with a specified scale.
~~~
def scaler_team(df_team, df, scaler):
    value = []
    for i in df_team.index:
        value.append(df.loc[i]['Margin'])
    value = np.array(value).reshape(-1, 1)
    scaled = scaler.transform(value)
    return np.mean(scaled)
~~~
~~~
def scaling(scaler):
    wickets_np = df_wickets['Margin'].to_numpy().reshape(-1, 1)
    scaler.fit(wickets_np)
    .
    .
    .
    return wicket_scaler, runs_scaler    
~~~
# Algorithm
We used a total of four algorithms: __Linear regression__, __logistic regression__, __random forest__, and __knn__.    
We used __linear regression__ first. However, linear regression showed approximately 45% accuracy.
   
We thought about the reason why the prediction result of linear regression is not good. The reason why the results of using linear regression were not good was that __there were two categories: winning or losing__. Therefore, we used __logistic regression__ that is more appropriate for categorical data than linear regression.
   
accuracy result:   
![image](https://user-images.githubusercontent.com/63892688/123589710-924eeb80-d824-11eb-84ac-593a688e4e2f.png)   
![image](https://user-images.githubusercontent.com/63892688/123589772-aabf0600-d824-11eb-8f5d-9af7adc3bb50.png)   
As a result of changing the algorithm, the accuracy has increased from about 45% to about 71%.   
   
After that, we have created a function to find the best combination that predicts the best results.
~~~
def best_combi(df_teams_minmax, df_teams_standard, df_teams_robust):
    return max(best_kfold_reg(df_teams_minmax), max(best_kfold_reg(df_teams_standard), best_kfold_reg(df_teams_robust)))
~~~
As a result, we found that combination with MinMaxScaler and Kfold of Logistic regression predicted the highest accuracy of 94%.   
![image](https://user-images.githubusercontent.com/63892688/123591342-cc20f180-d826-11eb-84e4-aa9cab7c43be.png)   
![image](https://user-images.githubusercontent.com/63892688/123591392-dd69fe00-d826-11eb-936c-89289bffd044.png)   

Therefore, we predicted the result of the game using the best combination we got earlier.    
Using the dataset of ic_rankings and fixtures, we printed out the winners for each game and found that 37 of the total 45 results were correct.   
![image](https://user-images.githubusercontent.com/63892688/123594462-b7465d00-d82a-11eb-9145-86d1ffe99cf3.png)
