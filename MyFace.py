import matplotlib as plt
import numpy as np
import sklearn as sk

def parse_data():

    # read data from source files
    posts_train = np.genfromtxt("posts_train.txt", delimiter = ",", skip_header=1)
    posts_test = np.genfromtxt("posts_test.txt", delimiter = ",", skip_header=1)
    friends = np.genfromtxt("small_graph.txt")

    return posts_train, posts_test, friends


def create_features(posts, friends, locations):
    pass
    # we should include some features from the posts of the friends of each person. Number, avg location, popularity, etc

def setup_data():

    # read the data in from the source files
    posts_train, posts_test, friends = parse_data()
    # locations = Ids with lat and lon columns from posts_train
    locations = np.concatenate((posts_train[:,0:1], posts_train[:, 4:6]), 1)
    # removes lat and lon from posts_train so it looks like test data
    posts_train = np.concatenate((posts_train[:,0:4], posts_train[:, 6:]), 1)

    # extract features from post and friend data
    features_train = create_features(posts_train, friends, locations)
    features_test = create_features(posts_test, friends, locations)
    # strip the Ids off of the training targets so they are just lat and lon
    targets_train = locations[:, 1:]

    return features_train, features_test, targets_train

if __name__ == "__main__":

    # parse our data into features and targets
    features_train, features_test, targets_train = setup_data()

    # train the learner
    learner = sk.neighbors.KNeighborsRegressor()
    learner.fit(features_train, targets_train)

    #predict
    predictions = (learner.predict(features_test))







