import numpy as np

#Test the data parsing functionality from MyFace.py
#Could put this into real unit testing form but I'm too lazy rn
if __name__ == "__main__":
    posts_train = np.genfromtxt("small_dataset.txt", delimiter = ",", skip_header=1)
    # locations = Ids with lat and lon columns from posts_train
    locations = np.concatenate((posts_train[:,0:1], posts_train[:, 4:6]), 1)
    # removes lat and lon from posts_train so it looks like test data
    posts_train = np.concatenate((posts_train[:,0:4], posts_train[:, 6:]), 1)

    print(posts_train)
    print(locations)