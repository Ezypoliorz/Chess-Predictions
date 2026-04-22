# fetch_data.py - Downloads the games data and merges it into JSON files

import requests
import zipfile
import io
import chess.pgn
import json
import multiprocessing

# Fetches the games data from The Week in Chess database (https://theweekinchess.com/twic)
def processIssue(issueNumber) :
    issueGames = []
    issuePlayers = {}
    
    # Defines the headers needed so send the request
    requestHeaders = {
        'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Referer' : 'https://theweekinchess.com/twic'
    }
    
    url = f"https://theweekinchess.com/zips/twic{issueNumber}g.zip"
    
    try :
        response = requests.get(url, headers=requestHeaders, timeout=30)
        if response.status_code != 200 :
            return [], {}
        
        # The downloaded files are Zip files
        # De-compresses those Zip files and opens the PGN inside
        with zipfile.ZipFile(io.BytesIO(response.content)) as z :
            pgnFilename = [name for name in z.namelist() if name.endswith('.pgn')][0]
            with z.open(pgnFilename) as f :
                pgnText = io.TextIOWrapper(f, encoding='latin-1', errors='replace')
                while True :
                    game = chess.pgn.read_game(pgnText)
                    if game is None :
                        break
                    
                    # Stores the basic game data
                    headers = game.headers
                    whiteName = headers.get('White')
                    blackName = headers.get('Black')
                    result = headers.get('Result')
                    eventName = headers.get('Event', '').lower()
                    siteName = headers.get('Site', '').lower()

                    whiteEloStr = headers.get('WhiteElo', '0')
                    blackEloStr = headers.get('BlackElo', '0')
                    
                    try :
                        whiteElo = int(whiteEloStr) if whiteEloStr.isdigit() else 0
                        blackElo = int(blackEloStr) if blackEloStr.isdigit() else 0
                    except (ValueError, TypeError) :
                        whiteElo, blackElo = 0, 0

                    # Removes as many games played by computers as possible
                    isComputer = (
                        headers.get('WhiteIsComputer') == 'Yes' or 
                        headers.get('BlackIsComputer') == 'Yes' or
                        any(engine in whiteName for engine in ['Stockfish', 'LC0', 'Leela', 'Komodo', 'Houdini']) or
                        any(engine in blackName for engine in ['Stockfish', 'LC0', 'Leela', 'Komodo', 'Houdini']) or
                        whiteElo > 2882 or
                        blackElo > 2882
                    )
                    
                    if isComputer :
                        continue

                    # Removes as many rapid and blitz games as possible
                    if any(keyword in eventName for keyword in ['blitz', 'rapid', 'online', 'bullet', 'arena', 'speed']) :
                        continue

                    # Removes online games from the dataset
                    # For some reason, lowers the accuracy considerably
                    """if any(keyword in siteName for keyword in ['chess.com', 'lichess']) :
                        continue"""
                    
                    if whiteElo == 0 or blackElo == 0 :
                        continue

                    for name, elo in [(whiteName, whiteElo), (blackName, blackElo)] :
                        if name not in issuePlayers :
                            issuePlayers[name] = {
                                'elos' : [], 
                                'games' : [], 
                                'wins' : 0, 
                                'draws' : 0, 
                                'losses' : 0
                            }
                        issuePlayers[name]['elos'].append(elo)
                        issuePlayers[name]['games'].append({
                            'whiteName' : whiteName,
                            'blackName' : blackName,
                            'result' : result
                        })
                    
                    if result == "1-0" :
                        issuePlayers[whiteName]['wins'] += 1
                        issuePlayers[blackName]['losses'] += 1
                    elif result == "0-1" :
                        issuePlayers[whiteName]['losses'] += 1
                        issuePlayers[blackName]['wins'] += 1
                    elif result == "1/2-1/2" :
                        issuePlayers[whiteName]['draws'] += 1
                        issuePlayers[blackName]['draws'] += 1

                    issueGames.append({
                        'date' : headers.get('Date'),
                        'whiteName' : whiteName,
                        'blackName' : blackName,
                        'whiteElo' : whiteElo,
                        'blackElo' : blackElo,
                        'result' : result,
                    })

        print(f"Finished processing issue {issueNumber}")
        return issueGames, issuePlayers

    except Exception as e :
        print(f"Error on TWIC {issueNumber} : {e}")
        return [], {}

# Downloads the PGNs using multiprocessing, and merges the data
def fetch(startIssue : int, endIssue : int, trainingSet : bool) :
    allGamesData = []
    globalPlayersDict = {}

    print(f"Starting parallel processing with {multiprocessing.cpu_count()} cores...")

    issues = range(startIssue, endIssue + 1)

    # Creates a pool of processes for multiprocessing
    with multiprocessing.Pool() as pool :
        results = pool.map(processIssue, issues)

    # Merges the data into a single list / dictionnary
    for issueGames, issuePlayers in results :
        allGamesData.extend(issueGames)
        for playerName, playerData in issuePlayers.items() :
            if playerName not in globalPlayersDict :
                globalPlayersDict[playerName] = {
                    'elos' : [], 
                    'games' : [], 
                    'wins' : 0, 
                    'draws' : 0, 
                    'losses' : 0
                }
            
            globalPlayersDict[playerName]['elos'].extend(playerData['elos'])
            globalPlayersDict[playerName]['games'].extend(playerData['games'])
            globalPlayersDict[playerName]['wins'] += playerData['wins']
            globalPlayersDict[playerName]['draws'] += playerData['draws']
            globalPlayersDict[playerName]['losses'] += playerData['losses']

    finalPlayersData = []
    for name, data in globalPlayersDict.items() :
        eloList = data['elos']
        totalGames = data['wins'] + data['draws'] + data['losses']

        if totalGames > 0 :
            finalPlayersData.append({
                'name' : name,
                'eloAverage' : int(sum(eloList) / len(eloList)) if eloList else 0,
                'winRate' : int(data['wins'] / totalGames * 100),
                'drawRate' : int(data['draws'] / totalGames * 100),
                'lossesRate' : int(data['losses'] / totalGames * 100),
                'games' : data['games']
            })

    if trainingSet == True :
        # Writes the data into the corresponding JSON files
        with open("training_players_data.json", 'w', encoding='latin-1') as playersFile :
            json.dump(finalPlayersData, playersFile, indent=4)

        with open("training_games_data.json", 'w', encoding='latin-1') as gamesFile :
            json.dump(allGamesData, gamesFile, indent=4)
    else :
       with open("testing_games_data.json", 'w', encoding='latin-1') as gamesFile :
            json.dump(allGamesData, gamesFile, indent=4) 

    print(f"Done ! {len(allGamesData)} games fetched and processed.")