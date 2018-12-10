import matplotlib as plt
import numpy as np
import sklearn as sk
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor as knnregressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE

def parse_data(posts_train_file, posts_test_file, graph_file):

    # read data from source files
    posts_train = np.genfromtxt(posts_train_file, delimiter = ",", skip_header=1)
    posts_test = np.genfromtxt(posts_test_file, delimiter = ",", skip_header=1)
    friends = np.genfromtxt(graph_file)

    return posts_train, posts_test, friends

def parse_data_train(posts_train_file, graph_file):

    # read data from source files
    posts_train = np.genfromtxt(posts_train_file, delimiter = ",", skip_header=1)
    friends = np.genfromtxt(graph_file)

    return posts_train, friends


def create_features(posts, friends, locations):

    extracted_features = []
    # compute popularity, avg location and avg sd of friend location
    for datapoint in posts:
        userId = datapoint[0]
        num_friends = len(friends[userId])
        avg_friend_lat, avg_friend_lon, sd_friend_lat, sd_friend_lon = create_friend_location_data(friends[userId], locations)
        #avg_friend_popularity = None
        extracted_features.append([num_friends, avg_friend_lat, avg_friend_lon, sd_friend_lat, sd_friend_lon])

    features = np.concatenate((posts[:, 1:], extracted_features), 1)
    return features

def create_friend_location_data(user_friends, locations):
    friend_lats = []
    friend_lons = []
    for friendId in user_friends:
        if friendId in locations:
            friend_lat = locations[friendId][0]
            friend_lon = locations[friendId][1]
            if friend_lat is not 0 and friend_lon is not 0:
                friend_lats.append(friend_lat)
                friend_lons.append(friend_lon)
    if len(friend_lons) > 0 and len(friend_lats) > 0:
        avg_lat = np.average(friend_lats)
        avg_lon = np.average(friend_lons)
        sd_lat = np.std(friend_lats)
        sd_lon = np.std(friend_lons)
        return avg_lat, avg_lon, sd_lat, sd_lon
    else:
        return 0,0,0,0

def create_friends_dict(friends_list):
    friends = defaultdict(list)
    for i in range(0,len(friends_list)):
        friends[friends_list[i][0]].append(friends_list[i][1])
    return friends

def create_locations_dict(ids, locations_list):
    locations = defaultdict(list)
    for i in range(0, len(locations_list)):
        locations[ids[i][0]] = locations_list[i]
    return locations


def setup_data_train():

    # read the data in from the source files
    all_posts, friends = parse_data_train("posts_train.txt", "graph.txt")

    posts_train = []
    posts_test = []

    for datapoint in all_posts:
        rand = np.random.randint(100)
        if rand is 0:
            posts_test.append(datapoint)
        else:
            posts_train.append(datapoint)

    posts_train = np.asarray(posts_train)
    posts_test = np.asarray(posts_test)

    # this code is just for testing the sample set rn
    targets_test = posts_test[:,4:6]
    posts_test = np.concatenate((posts_test[:,0:4], posts_test[:, 6:]), 1)

    # create friends dictionary
    friends = create_friends_dict(friends)

    # this is duplicate data with locations, but keeping it for now for ease
    targets_train = posts_train[:, 4:6]
    # locations = Ids with lat and lon columns from posts_train
    locations = create_locations_dict(posts_train[:,0:1], posts_train[:, 4:6])
    # removes lat and lon from posts_train so it looks like test data
    posts_train = np.concatenate((posts_train[:,0:4], posts_train[:, 6:]), 1)
    # extract features from post and friend data
    features_train = create_features(posts_train, friends, locations)
    features_test = create_features(posts_test, friends, locations)

    return features_train, features_test, targets_train, targets_test

def setup_data_test():
    print("Parsing Data \n")
    # read the data in from the source files
    posts_train, posts_test, friends = parse_data("posts_train.txt", "posts_test.txt", "graph.txt")

    print("Creating Friends Dictionary \n")
    # create friends dictionary
    friends = create_friends_dict(friends)

    # this is duplicate data with locations, but keeping it for now for ease
    targets_train = posts_train[:, 4:6]
    # locations = Ids with lat and lon columns from posts_train
    locations = create_locations_dict(posts_train[:,0:1], posts_train[:, 4:6])
    # removes lat and lon from posts_train so it looks like test data
    posts_train = np.concatenate((posts_train[:,0:4], posts_train[:, 6:]), 1)

    test_ids = posts_test[:, [0]]
    print(test_ids.shape)
    print("Extracting Features \n")
    # extract features from post and friend data
    features_train = create_features(posts_train, friends, locations)
    features_test = create_features(posts_test, friends, locations)

    print("Done Setting Up Data \n")
    return features_train, features_test, targets_train, test_ids

def train_learner(features_train, targets_train):
    # train the learner
    learner = knnregressor(weights='distance', n_neighbors=20)
    learner.fit(features_train, targets_train)
    return learner

def run_learner_train():
    # parse our data into features and targets
    features_train, features_test, targets_train, targets_test = setup_data_train()

    #train the learner
    learner = train_learner(features_train, targets_train)
    #print(learner.score(features_test, targets_test))

    #predict
    predictions = (learner.predict(features_test))
    print(predictions)
    print(targets_test)
    print(np.sqrt(MSE(predictions, targets_test)))

def run_learner_test():
    # parse our data into features and targets
    features_train, features_test, targets_train, test_ids = setup_data_test()

    print("Training Learner")
    #train the learner
    learner = train_learner(features_train, targets_train)
    #print(learner.score(features_test, targets_test))

    print("Predicting")
    #predict
    predictions = (learner.predict(features_test))
    predictions = np.concatenate((test_ids, predictions), 1)
    predictions_file = open("predictions_2.txt", "w")
    np.savetxt(predictions_file, predictions, fmt="%d,%f,%f", delimiter=",", header="Id,Lat,Lon")
    predictions_file.close()

def grid_search(learner, params):
    #Search through different paramater combinations for a learner
    X_tr, X_te, y_tr, y_te = setup_data_train()
    n = GridSearchCV(learner, params, cv=5, scoring='neg_mean_squared_error')
    n.fit(X_tr, y_tr)
    best = n.best_estimator_
    print(best)
    y_pr = best.predict(X_te)
    print(y_pr)
    print(y_te)
    print(np.sqrt(MSE(y_te, y_pr)))

if __name__ == "__main__":
    #print("No Grid Search: \n\n")
    run_learner_test()
    #run_learner_train()
    #print("Grid Search: \n\n")
    #learner = knnregressor()
    #params = {'n_neighbors': (3, 10, 20, 30), 'weights': ('uniform', 'distance')}
    #grid_search(learner, params)






