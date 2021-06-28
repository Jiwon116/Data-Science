#  Data Science
#  Term project - Cricket World Cup Matches Prediction
#  Name            :  Jo Byeong Geun, Choi Ji Won, Jeong Da Hee
#  Student Number  :  201835528, 201835538, 201931889
#  Class           :  Mon 5678
#  Date            :  2021.05.26
#  Update          :  2021.06.11

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm, datasets
from warnings import simplefilter
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

simplefilter(action='ignore', category=FutureWarning)

# Make Confusion Matirx
def make_cf_matrix(y_test, y_pred):
    # Compare y testing sets with predicted y values
    cf_matrix = confusion_matrix(y_test, y_pred)
    # Show the confusion Matrix
    print("\n***************Confusion Matrix******************")
    print(cf_matrix)

# Make Classification Report
def make_cf_report(y_test, y_pred):
    # Evaluate the precision and reproducibility
    cf_report = classification_report(y_test, y_pred)
    # Show the Classificatin Report
    print("\n***************Classification Report******************")
    print(cf_report)

# Scaling data from wicketts, runs
def scaler_team(df_team, df, scaler):
    value = []
    for i in df_team.index:
        # Importing data in the 'Margin' column
        value.append(df.loc[i]['Margin'])
    # Change dimension and shape
    value = np.array(value).reshape(-1, 1)
    scaled = scaler.transform(value)
    # Scaling to mean values
    return np.mean(scaled)

# Scaling using scaler
def scaling(scaler):
    wickets_np = df_wickets['Margin'].to_numpy().reshape(-1, 1)
    scaler.fit(wickets_np)

    # Declaring a list to insert values for scaling
    wicket_scaler = []
    wicket_scaler.append(scaler_team(df_England_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_SouthAfrica_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_WestIndies_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_Pakistan_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_NewZealand_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_SriLanka_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_Afghanistan_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_Australia_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_Bangladesh_wicket, df_wickets, scaler))
    wicket_scaler.append(scaler_team(df_India_wicket, df_wickets, scaler))

    runs_np = df_runs['Margin'].to_numpy().reshape(-1, 1)
    scaler.fit(runs_np)
    # Declaring a list to insert values for scaling
    runs_scaler = []
    runs_scaler.append(scaler_team(df_England_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_SouthAfrica_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_WestIndies_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_Pakistan_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_NewZealand_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_SriLanka_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_Afghanistan_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_Australia_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_Bangladesh_runs, df_runs, scaler))
    runs_scaler.append(scaler_team(df_India_runs, df_runs, scaler))

    # Outputs scaling results for worldcup_team in bar form
    plt.bar(worldcup_teams, wicket_scaler)
    # X axis name
    plt.xlabel('Teams')
    # Y axis name
    plt.ylabel('Scaled Wicket')
    # Show plotting result
    plt.show()
    # Outputs scaling results for worldcup_team in bar form
    plt.bar(worldcup_teams, runs_scaler)
    # X axis name
    plt.xlabel('Teams')
    # Y axis name
    plt.ylabel('Scaled Runs')
    # Show plotting result
    plt.show()

    return wicket_scaler, runs_scaler

# Kfold- Linear Regression
def kfold_lin(kfold, df_teams):
    sum_training = 0
    sum_testing = 0
    count = 0
    for train_index, test_index in kfold.split(df_teams):
        # Data for training
        df_train = df_teams.iloc[train_index]
        # Data for testing
        df_test = df_teams.iloc[test_index]
        model = LinearRegression()
        # Change the training set and testing set to the correct structure and store them.
        train = pd.get_dummies(df_train, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
        test = pd.get_dummies(df_test, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
        # Drop 'Winner' column of training set and put it in the X training set.
        X_train = train.drop(['Winner'], axis=1)
        # Insert the 'Winner' column of training set into the y training set.
        y_train = train["Winner"]
        # Drop 'Winner' column of testing set and put it in the X testing set.
        X_test = test.drop(['Winner'], axis=1)
        # Insert the 'Winner' column of testing set into the y testing set.
        y_test = test["Winner"]
        model.fit(X_train, y_train)
        # Derives the value of learning the training set to Linear Regression
        score = model.score(X_train, y_train)
        # Derives the value of learning the testing set to Linear Regression
        score2 = model.score(X_test, y_test)
        count = count + 1
        sum_training = sum_training + score
        sum_testing = sum_testing + score2
        # Accuracy output each K-fold data
        print("Training set accuracy: ", '%.3f' % (score))
        print("Test set accuracy: ", '%.3f' % (score2))
    print("Training set accuracy average: ", '%.3f' % (sum_training / count))
    print("Test set accuracy average: ", '%.3f' % (sum_testing / count))
    return sum_testing/count

# Kfold - Logistic Regression
def kfold_log(kfold,df_teams):
    sum_training = 0
    sum_testing = 0
    count = 0
    for train_index, test_index in kfold.split(df_teams):
        # Data for training
        df_train = df_teams.iloc[train_index]
        # Data for testing
        df_test = df_teams.iloc[test_index]
        model = LogisticRegression()
        # Change the training set and testing set to the correct structure and store them.
        train = pd.get_dummies(df_train, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
        test = pd.get_dummies(df_test, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
        # Drop 'Winner' column of training set and put it in the X training set.
        X_train = train.drop(['Winner'], axis=1)
        # Insert the 'Winner' column of training set into the y training set.
        y_train = train["Winner"]
        # Drop 'Winner' column of testing set and put it in the X testing set.
        X_test = test.drop(['Winner'], axis=1)
        # Insert the 'Winner' column of testing set into the y testing set.
        y_test = test["Winner"]
        model.fit(X_train, y_train)
        # Derives the value of learning the training set to Logistic Regression
        score = model.score(X_train, y_train)
        # Derives the value of learning the testing set to Logistic Regression
        score2 = model.score(X_test, y_test)
        count = count + 1
        sum_training = sum_training + score
        sum_testing = sum_testing + score2
        # Accuracy output each K-fold data
        print("Training set accuracy: ", '%.3f' % (score))
        print("Test set accuracy: ", '%.3f' % (score2))
    print("Training set accuracy average: ", '%.3f' % (sum_training / count))
    print("Test set accuracy average: ", '%.3f' % (sum_testing / count))
    return sum_testing/count

# Kfold - Random Forest
def kfold_ran(kfold, best_score, df_teams):
    sum_training=0
    sum_testing=0
    count=0
    for train_index, test_index in kfold.split(df_teams):
        # Data for training
        df_train = df_teams.iloc[train_index]
        # Data for testing
        df_test = df_teams.iloc[test_index]
        # Change the training set and testing set to the correct structure and store them.
        train = pd.get_dummies(df_train, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
        test = pd.get_dummies(df_test, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

        # Drop 'Winner' column of training set and put it in the X training set.
        X_train = train.drop(['Winner'], axis=1)
        # Insert the 'Winner' column of training set into the y training set.
        y_train = train["Winner"]
        # Drop 'Winner' column of testing set and put it in the X testing set.
        X_test = test.drop(['Winner'], axis=1)
        # Insert the 'Winner' column of testing set into the y testing set.
        y_test = test["Winner"]
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
        # Learning X training set by y training
        rf.fit(X_train, y_train)
        # Derives the value of learning the training set to Random Forest
        score = rf.score(X_train, y_train)
        # Derives the value of learning the testing set to Random Forest
        score2 = rf.score(X_test, y_test)
        count=count+1
        sum_training=sum_training+score
        sum_testing=sum_testing+score2

        # Store the Random Forest of the best test value that have learned
        if best_score < score2:
            best_score = score2
            best_rf = rf

        # Accuracy output each K-fold data
        print("Training set accuracy: ", '%.3f' % (score))
        print("Test set accuracy: ", '%.3f' % (score2))


    print("Training set accuracy average: ", '%.3f' % (sum_training/count))
    print("Test set accuracy average: ", '%.3f' % (sum_testing/count))
    return sum_testing/count, best_rf


# Label Encoding
def labelEncoding(df_teams):
        df_teams = df_teams.apply(LabelEncoder().fit_transform)
        return df_teams

# Best accuracy of K-Fold & algorithm (Linear, Logistic and RandomForest)
def best_kfold_reg(df_teams):
    #K-fold Linear Regression
    print("\n*************Kfold of Linear Regression***************")
    scores = []
    kfold = KFold(n_splits=3, shuffle=True)
    scaled_kfold_lin = kfold_lin(kfold, df_teams)

    #K-fold Logistic Regression
    print("\n*************Kfold of Logistic Regression***************")
    scores = []
    kfold = KFold(n_splits=3, shuffle=True)
    df_teams['Winner']=df_teams['Winner'].astype(int)
    scaled_kfold_log = kfold_log(kfold,df_teams)

    # Kfold random forest
    # Split data into three
    print("\n*************Kfold of Random Forest ***************")  
    kfold = KFold(n_splits=3, shuffle=True)
    # Declaring a variable to put the best value in
    best_score = 0
    scaled_kfold_ran, best_rf = kfold_ran(kfold, best_score, df_teams)

    best_acc = bestAcc(scaled_kfold_lin, scaled_kfold_log, scaled_kfold_ran)
    #print("\nthe best acc = ", '%0.3f' % best_acc)

    return best_acc

# Calculate best accuracy
def bestAcc(lin, log, ran):
    return max(lin, max(log, ran))

# Best combination
def best_combi(df_teams_minmax, df_teams_standard, df_teams_robust):
    return max(best_kfold_reg(df_teams_minmax), max(best_kfold_reg(df_teams_standard), best_kfold_reg(df_teams_robust)))




# Read the csv file
results = pd.read_csv('dataset/results.csv')

# dataset statistical data
print("\n***********statistical data***************")
print(results.describe())
# Feature names
print("\n***********feature names***************")
print(results.columns.values)
# data types
print("\n***********feature data types***************")
print(results.dtypes)
# data shape
print("\n***********data shape***************")
print(results.shape)
# data index
print("\n***********data index***************")
print(results.index)
# dataset columns
print("\n***********dataset columns***************")
print(results.columns)
# Top 5 data in dataset
print("\n***********dataset***************")
print(results.head())

# data preprocessing
# Drop missing null or ground information that is not needed
results.dropna(axis=0, inplace=True)
results.drop(['Ground'], axis=1, inplace=True)
# Limited to 10 teams participating in the World Cup
worldcup_teams = ['England', 'South Africa', 'West Indies',
                  'Pakistan', 'New Zealand', 'Sri Lanka', 'Afghanistan',
                  'Australia', 'Bangladesh', 'India']

# Include only countries corresponding to worldcup_teams in results dataset of column 'Team_1'
df_teams_1 = results[results['Team_1'].isin(worldcup_teams)]
# Include only countries corresponding to worldcup_teams in results dataset of column 'Team_2'
df_teams = df_teams_1[df_teams_1['Team_2'].isin(worldcup_teams)]
# deduplication
df_teams = df_teams.drop_duplicates()
# Clean up index
df_teams = df_teams.sort_index(ascending=True)

# Show df_teams dataset
print("\n***********df_teams dataset***************")
print(df_teams.head())
# data shape
print("\n***********df_teams data shape***************")
print(df_teams.shape)

# If the defense team wins
# data with a value of 'Wicket' in the 'Margin' column will be separately saved.
df_wickets = df_teams.loc[df_teams['Margin'].str.contains('wickets')].copy()
# To omit the word 'wickets' from the value and only insert numbers (int types)
df_wickets['Margin'] = df_wickets.Margin.str.extract('(\d+)')
# Create time series data using the 'data' value
df_wickets['year'] = pd.DatetimeIndex(df_wickets['date']).year

df_England_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[0]]
df_SouthAfrica_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[1]]
df_WestIndies_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[2]]
df_Pakistan_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[3]]
df_NewZealand_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[4]]
df_SriLanka_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[5]]
df_Afghanistan_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[6]]
df_Australia_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[7]]
df_Bangladesh_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[8]]
df_India_wicket = df_wickets[df_wickets['Winner'] == worldcup_teams[9]]


# If the offense team wins
# data with a value of 'runs' in the 'Margin' column will be separately saved.
df_runs = df_teams.loc[df_teams['Margin'].str.contains('runs')].copy()
# To omit the word 'runs' from the value and only insert numbers (int types)
df_runs['Margin'] = df_runs.Margin.str.extract('(\d+)')
# Create time series data using the 'data' value
df_runs['year'] = pd.DatetimeIndex(df_runs['date']).year

df_England_runs = df_runs[df_runs['Winner'] == worldcup_teams[0]]
df_SouthAfrica_runs = df_runs[df_runs['Winner'] == worldcup_teams[1]]
df_WestIndies_runs = df_runs[df_runs['Winner'] == worldcup_teams[2]]
df_Pakistan_runs = df_runs[df_runs['Winner'] == worldcup_teams[3]]
df_NewZealand_runs = df_runs[df_runs['Winner'] == worldcup_teams[4]]
df_SriLanka_runs = df_runs[df_runs['Winner'] == worldcup_teams[5]]
df_Afghanistan_runs = df_runs[df_runs['Winner'] == worldcup_teams[6]]
df_Australia_runs = df_runs[df_runs['Winner'] == worldcup_teams[7]]
df_Bangladesh_runs = df_runs[df_runs['Winner'] == worldcup_teams[8]]
df_India_runs = df_runs[df_runs['Winner'] == worldcup_teams[9]]



########### Scaling ###########
# MinMax Scaling
minmax_scaler = MinMaxScaler()
minmax_wicket_scaler, minmax_runs_scaler = scaling(minmax_scaler)

# Standard Scaling
standard_scaler = StandardScaler()
standard_wicket_scaler, standard_runs_scaler = scaling(standard_scaler)

# Robust Scaling
robust_scaler = RobustScaler()
robust_wicket_scaler, robust_runs_scaler = scaling(robust_scaler)


# Drop columns of 'Margin' and 'date' that has nothing to do with winning or losing.
df_teams = df_teams.drop(['date', 'Margin'], axis=1)
# reorder index
df_teams = df_teams.reset_index(drop=True)

########## Encoding ###########
# LabelEncoding to change team name to number
df_teams = labelEncoding(df_teams)

# Divide the categorical properties for Team_1 and Team_2.
final = pd.get_dummies(df_teams, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
# Drop 'Winner' column from final dataset and put it in X
X = final.drop(['Winner'], axis=1)
# Store only 'Winner' column in y in final dataset
y = final["Winner"]
# Divide training and testing set by 7:3 each.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
# Learning X training set by y training
rf.fit(X_train, y_train)
# Predicting an X testing set with learning results
y_pred = rf.predict(X_test)
# Derives the value of learning the training set to Random Forest
score = rf.score(X_train, y_train)
# Derives the value of learning the testing set to Random Forest
score2 = rf.score(X_test, y_test)
# Show Random Forest
print("\n***************Random Forest******************")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Accuracy output
print("Training set accuracy: ", '%.3f' % (score))
print("Test set accuracy: ", '%.3f' % (score2))

# K neareast neighbor
k_list = range(1,10)
training_accuracy = []
test_accuracy = []
for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  # Learning X training set by y training
  classifier.fit(X_train, y_train)
  # Store accuracy of training set
  training_accuracy.append(classifier.score(X_train, y_train))
  # Store accuracy of testing set
  test_accuracy.append(classifier.score(X_test, y_test))

# Outputs K neareast neighbor result
plt.plot(k_list, training_accuracy)
# X axis name
plt.xlabel("n_neighbors")
# Y axis name
plt.ylabel("Accuracy")
# Show plotting result
plt.show()

# Prediction of testing set
clf = KNeighborsClassifier(n_neighbors = 7)
# Learning X training set by y training
clf.fit(X_train, y_train)
# Show the prediction and accuracy of testing set
print("\n*********************KNN***********************")
print("Test Prediction: {}".format(clf.predict(X_test)))
print("Test Accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# Linear Regression
linreg = LinearRegression()
# Learning X training set by y training
linreg.fit(X_train, y_train)
# Predicting an X testing set with learning results
y_pred = linreg.predict(X_test)
# Derives the value of learning the training set to Linear regression
score = linreg.score(X_train, y_train)
# Derives the value of learning the testing set to Linear regression
score2 = linreg.score(X_test, y_test)
# Show Linear Regression
print("\n***************Linear Regression******************")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Accuracy output
print("Training set accuracy: ", '%.3f' % (score))
print("Test set accuracy: ", '%.3f' % (score2))


# Logistic Regression
logreg = LogisticRegression()
# Learning X training set by y training
logreg.fit(X_train, y_train)
# Predicting an X testing set with learning results
y_pred = logreg.predict(X_test)
# Derives the value of learning the training set to Logistic regression
score = logreg.score(X_train, y_train)
# Derives the value of learning the testing set to Logistic regression
score2 = logreg.score(X_test, y_test)
# Show Logistic Regression
print("\n******************Logistic Regression*********************")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Accuracy output
print("Training set accuracy: ", '%.3f' % (score))
print("Test set accuracy: ", '%.3f' % (score2))
# Show Confusion Matrix and Classification Report
make_cf_matrix(y_test, y_pred)
make_cf_report( y_test, y_pred)


# Confusion matrix Visualization
cf_matrix = confusion_matrix(y_test, y_pred)
# Support Vector classifier
clf = svm.SVC(random_state=0)
# Learning X training set by y training
clf.fit(X_train, y_train)
svm.SVC(random_state=0)
# Decide how to show the matrix
matrix = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
# Title of matrix table
plt.title('Confusion matrix for our classifier')
# Show the matrix table
plt.show()


Scaler = MinMaxScaler()
df_teams_minmax = pd.DataFrame(Scaler.fit_transform(df_teams), columns=['Team_1', 'Team_2', 'Winner'])
print(df_teams_minmax)

Scaler = StandardScaler()
df_teams_standard = pd.DataFrame(Scaler.fit_transform(df_teams), columns=['Team_1', 'Team_2', 'Winner'])
print(df_teams_standard)

Scaler = RobustScaler()
df_teams_robust = pd.DataFrame(Scaler.fit_transform(df_teams), columns=['Team_1', 'Team_2', 'Winner'])
print(df_teams_robust)

best_combi_acc = best_combi(df_teams_minmax, df_teams_standard, df_teams_robust)
print("\nbest combination accuracy : ", best_combi_acc)

# Expected qualifier rank
# Read the csv file
ranking = pd.read_csv('dataset/icc_rankings.csv')
fixtures = pd.read_csv('dataset/fixtures.csv')
# Show Top 5 data of fixtures dataset
print("\n***************icc_rankings dataset******************")
print(ranking.head())
# Show last 5 data of fixtures dataset
print("\n***************Fixtures dataset******************")
print(fixtures.tail())

# Ranked using icc_ranking dataset
fixtures.insert(1, 'first_position', fixtures['Team_1'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Team_2'].map(ranking.set_index('Team')['Position']))
# Find the value through the integral position
fixtures = fixtures.iloc[:45, :]
# Show last 5 data of fixtures dataset with ranking
print("\n***************fixtures dataset with ranking******************")
print(fixtures.tail())

# Declaring a variable to put the predicted value
pred_ranking_set = []
# The first variable is indexed, and the second variable is accessed one by one in the row in the column.
for idx, row in fixtures.iterrows():
    # Change the value of team_1 or team_2 according to rank
    if row['first_position'] < row['second_position']:
        pred_ranking_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'Winner': None})
    else:
        pred_ranking_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'Winner': None})
# Convert DataFrame type
pred_ranking_set = pd.DataFrame(pred_ranking_set)
# Top 5 data in dataset
print("\n***************Predicted dataset******************")
print(pred_ranking_set.head())

# Declares variables for backing up data in 'df_teams'
backup_df_teams = df_teams
# Divide the categorical properties for Team_1 and Team_2.
backup_final = pd.get_dummies(backup_df_teams, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
# Declares variables for backing up data in 'pred_set'
backup_pred_set = pred_ranking_set

# Divide the categorical properties for Team_1 and Team_2.
pred_ranking_set = pd.get_dummies(pred_ranking_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
# Store the differences between existing dataset and predicted dataset
missing_cols = set(backup_final.columns) - set(pred_ranking_set.columns)
# Store missing value as 0 in predicted dataset
for c in missing_cols:
    pred_ranking_set[c] = 0
# Insert data of the column in 'backup_final' into the predicted dataset
pred_ranking_set = pred_ranking_set[backup_final.columns]
# Drop the 'Winner' column
pred_ranking_set = pred_ranking_set.drop(['Winner'], axis=1)

# Put the dataset to experiment into the random forest to derive the result value
best_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
best_rf.fit(X_train, y_train)
predict_result = best_rf.predict(pred_ranking_set)
# Show expected qualifier rank
print("\n*************Qualifier***************")
for i in range(fixtures.shape[0]):
    # Show the country to compete with
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predict_result[i] == 1:
        # The second written country wins
        print("Winner: " + backup_pred_set.iloc[i, 1])
    else:
        # The first written country wins
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print("")