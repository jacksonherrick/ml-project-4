import matplotlib as plt
import numpy as np
import sklearn as sk

def parse_data():

    pass

def parse_friends():

    pass

def create_features(data, friends):
    pass

def predict_lat(features):
    pass
    #add predicted latitude to longitude features

def predict_long(features):
    pass

def parse_test_data():
    pass

if __name__ == "__main__":


    pass
    data, targets = parse_data()
    friends = parse_friends()
    features = create_features(data, friends)
    test_data = parse_test_data()

    learner = sk.neighbors.KNeighborsRegressor()
    learner.fit(features, targets)

    predictions = (learner.predict(test_data))







