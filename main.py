# main.py - Main program, executes all the other scripts in the correct order
# This should take a few minutes and give a 62% accuracy

import fetch_data
import train_model
import test_model
import time

def main() :
    print("Starting the training and evaluation process.")

    start_time = time.perf_counter()

    print("Calling fetching function.")
    fetch_data.fetch(startIssue=1589, endIssue=1639)
    print(f"Fetching took {int(time.perf_counter() - start_time)}s")
    intermediate_time = time.perf_counter()

    print("Calling training function.")
    train_model.trainModel()
    print(f"Training the model took {int(time.perf_counter() - intermediate_time)}s")
    intermediate_time = time.perf_counter()

    print("Calling evaluation function.")
    test_model.testModel(1636, 1637)
    print(f"Evaluating the PGN took {int(time.perf_counter() - intermediate_time)}s")
    print(f"Total took {int(time.perf_counter() - start_time)}s")

if __name__ == "__main__" :
    main()