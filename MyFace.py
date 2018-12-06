import matplotlib as plt
import numpy as np
import sklearn as sk
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor as knnregressor

def parse_data(posts_train_file, posts_test_file, graph_file):

    # read data from source files
    posts_train = np.genfromtxt(posts_train_file, delimiter = ",", skip_header=1)
    posts_test = np.genfromtxt(posts_test_file, delimiter = ",", skip_header=1)
    friends = np.genfromtxt(graph_file)

    return posts_train, posts_test, friends


def create_features(posts, friends, locations):
    pass
    # we should include some features from the posts of the friends of each person. Number, avg location, popularity, etc

    return posts

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


def setup_data():

    # read the data in from the source files
    posts_train, posts_test, friends = parse_data("sample_posts_train.txt", "sample_posts_test.txt", "small_graph.txt")

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

def train_learner(features_train, targets_train):
    # train the learner
    learner = knnregressor(weights='distance')
    learner.fit(features_train, targets_train)
    #print(learner.score(features_test, targets_test))
    return learner

def run_learner():
    # parse our data into features and targets
    features_train, features_test, targets_train, targets_test = setup_data()

    #train the learner
    #learner = train_learner(features_train, targets_train)
    #predict
    #predictions = (learner.predict(features_test))
    #print(predictions)
    #print(targets_test)


if __name__ == "__main__":
    run_learner()








