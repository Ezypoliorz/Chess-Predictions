# train_model.py - Trains the XGboost model on the data from the JSON files

import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def loadFeatures(jsonPath, playerMap) :
    with open(jsonPath, 'r', encoding = 'latin-1') as f :
        games = json.load(f)
    
    features = []
    labels = []

    # Maps the text results to integers
    resultMap = {'1-0' : 0, '1/2-1/2' : 1, '0-1' : 2}

    for game in games :
        white = playerMap.get(game['whiteName'])
        black = playerMap.get(game['blackName'])
        
        if not white or not black or game['result'] not in resultMap : 
            continue
    
        whiteElo = int(game['whiteElo'])
        blackElo = int(game['blackElo'])
        
        eloDiff = whiteElo - blackElo
        
        # Creates the features passed to the model
        row = [
            whiteElo, blackElo, eloDiff, 
            white['winRate'], white['drawRate'], white['lossesRate'],
            black['winRate'], black['drawRate'], black['lossesRate'],
        ]
        features.append(row)
        labels.append(resultMap[game['result']])
    
    return np.array(features), np.array(labels)

def trainModel() :
    print("Loading data for XGBoost training...")

    with open("training_players_data.json", 'r', encoding = 'latin-1') as f :
        trainingPlayersRaw = json.load(f)

    # Maps the players and their relevant data
    playerMap = {p['name'] : {'elo' : p.get('eloAverage', 2000), 'winRate' : p.get('winRate'), 'drawRate' : p.get('drawRate'), 'lossesRate' : p.get('lossesRate')} for p in trainingPlayersRaw}

    XTrain, yTrain = loadFeatures("training_games_data.json", playerMap)
    XTest, yTest = loadFeatures("testing_games_data.json", playerMap)

    # Define feature names for better visualization
    statsLabels = [
        "whiteElo", "blackElo", "eloDiff",
        "whiteWinRate", "whiteDrawRate", "whiteLossRate", 
        "blackWinRate", "blackDrawRate", "blackLossRate"
    ]
    allFeatureNames = statsLabels

    print(f"Training on {len(XTrain)} samples with {len(statsLabels)} features...")
    
    # Defines the parameters of the model
    model = xgb.XGBClassifier(
        objective = 'multi:softprob',
        num_class = 3,
        n_estimators = 200,
        max_depth = 8,
        learning_rate = 0.01,
        subsample = 0.8,
        colsample_bytree = 0.6,
        random_state = 42,
        eval_metric = ['mlogloss', 'merror'] # Tracking logloss and error rate
    )

    # Trains the model with evaluation set, to monitor progress
    evalSet = [(XTrain, yTrain), (XTest, yTest)]
    model.fit(
        XTrain, yTrain, 
        eval_set = evalSet, 
        verbose = False
    )

    # Plotting the graphs to vizualise
    results = model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    xAxis = range(0, epochs)
    
    plt.figure(figsize = (12, 5))
    
    # Plot Logloss
    plt.subplot(1, 2, 1)
    plt.plot(xAxis, results['validation_0']['mlogloss'], label = 'Train')
    plt.plot(xAxis, results['validation_1']['mlogloss'], label = 'Test')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss Evolution')

    # 2. Plot Feature Importance
    plt.subplot(1, 2, 2)
    # Mapping feature names to the model for the plot
    model.get_booster().feature_names = allFeatureNames
    xgb.plot_importance(model, max_num_features = 9, ax = plt.gca(), importance_type = 'weight')
    plt.title('Features importance')
    
    plt.tight_layout()
    plt.show()
    
    # Extracting predictions and probabilities for the complete report
    predictions = model.predict(XTest)
    probabilities = model.predict_proba(XTest)

    # Calculates the expected results
    expectedWhiteWins = np.sum(probabilities[:, 0])
    expectedDraws = np.sum(probabilities[:, 1])
    expectedBlackWins = np.sum(probabilities[:, 2])

    # Calculates the actual results
    actualWhiteWins = np.sum(yTest == 0)
    actualDraws = np.sum(yTest == 1)
    actualBlackWins = np.sum(yTest == 2)
    
    # Calculates the accuracy
    accuracy = (np.sum(predictions == yTest) / len(yTest)) * 100

    # Prints the merged results
    print("-" * 40)
    print(f"Test Results :")
    print(f"Games Analyzed : {len(yTest)}")
    print(f"Global Accuracy : {accuracy :.2f} %")
    print("-" * 40)
    print(f"{'Result' : <15} | {'Expected' : <10} | {'Actual' : <10}")
    print(f"{'-' * 40}")
    print(f"{'White Wins' : <15} | {expectedWhiteWins : <10.1f} | {actualWhiteWins : <10}")
    print(f"{'Draws' : <15} | {expectedDraws : <10.1f} | {actualDraws : <10}")
    print(f"{'Black Wins' : <15} | {expectedBlackWins : <10.1f} | {actualBlackWins : <10}")
    print("-" * 40)
    
    print("\nClassification Report :")
    print(classification_report(yTest, predictions))

    # Saves the model's parameters in a JSON file
    model.save_model("chess_prediction_model.json")
    print("Model saved.")

if __name__ == "__main__" :
    trainModel()