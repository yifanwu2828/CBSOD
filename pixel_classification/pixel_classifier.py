"""
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
"""

import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cv2
import generate_rgb_data as rgb_data

from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support

class PixelClassifier():
    def __init__(self,):
        self.red_classifer = LogisticRegression(color='red', pretrained=True)
        self.green_classifer = LogisticRegression(color='green', pretrained=True)
        self.blue_classifer = LogisticRegression(color='blue', pretrained=True)
        self.red_w = self.red_classifer.w
        self.green_w = self.green_classifer.w
        self.blue_w = self.blue_classifer.w

    def classify(self, X):
        red_linear = np.dot(X, self.red_w)
        green_linear = np.dot(X, self.green_w)
        blue_linear = np.dot(X, self.blue_w)
        y_pred_red = self.sigmoid(red_linear).reshape([1, -1])
        y_pred_green = self.sigmoid(green_linear).reshape([1, -1])
        y_pred_blue = self.sigmoid(blue_linear).reshape([1, -1])
        y_pred = np.vstack((y_pred_red, y_pred_green, y_pred_blue))
        y_pred_label = np.argmax(y_pred, axis=0)

        return y_pred_label

    def sigmoid(self, z):
        """
        sigmoid function
        :param z:
        :return: np.float64
        """
        return 1.0 / (1 + np.exp(-z))


class LogisticRegression():
    def __init__(self, color, pretrained=True):
        """
        Initialize classifier
        :param: color: which binary color classifier
        :param learning_rate: step size, default:1e-4
        :param max_iter: maximum num of iterations, default:1000
        """
        self.pretrained = pretrained
        if pretrained:
            self.load(color, fname='model.pkl')

    def classify(self, X):
        """
        Classify a set of pixels into red, green, or blue
        Inputs:
          X: n x 3 matrix of RGB values
        Outputs:
          y: n x 1 vector of with {-1,1} values corresponding to {not_blue, blue}, respectively
        """
        linear = np.dot(X, self.w)
        y_pred = self.sigmoid(linear)
        y_pred_label = [1 if i >= 0.5 else 0 for i in y_pred]

        return np.array(y_pred_label)

    def sigmoid(self, z):
        """
        sigmoid function
        :param z:
        :return: np.float64
        """
        return 1.0 / (1 + np.exp(-z))

    def fit(self, X, y, learning_rate=1e-2, max_iter=10000):
        """
        Batch Gradient Descent to find the optimal w
        :param X: training_set(r,g,b) with dim (1,3) -> np.array
        :param y: label of each training data either -1 or 1 ->np.array
        """
        assert len(np.unique(y)) == 2

        n_sample, n_dim = X.shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = np.random.randn(n_dim, 1)
        #self.w = np.zeros((n_dim, 1))
        y = y.reshape([-1, 1])

        for itr in range(self.max_iter + 1):
            linear = np.dot(X, self.w )
            y_pred = self.sigmoid(linear)
            dw = (1/n_sample) * np.dot(X.T, (y_pred-y))
            self.w -= self.learning_rate * dw
            # if itr % 200 == 0:
            #     print('after iteration {}:'.format(itr, ))
        return self.w

    def save(self, color, fname):
        assert isinstance(color, str)
        fname = color + 'model.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, color, fname):
        assert isinstance(color, str)
        fname = color + 'model.pkl'
        with open(fname, 'rb') as f:
            self.__dict__.update(pickle.load(f))


#######################
###                 ###
### Helper Function ###
###                 ###
#######################

def get_training_data(folder):
    """
    Helper function to get training data with label {1,2.3} -> {r,g,b}
    :param folder: folder contain the training img
    :return: X, y -> np.array
    """
    X1 = rgb_data.read_pixels(folder + '/red', verbose=False)
    X2 = rgb_data.read_pixels(folder + '/green')
    X3 = rgb_data.read_pixels(folder + '/blue')
    y1, y2, y3 = np.full(X1.shape[0], 1), np.full(X2.shape[0], 2), np.full(X3.shape[0], 3)
    X, y = np.concatenate((X1, X2, X3)), np.concatenate((y1, y2, y3))
    return X, y


def binary_label(raw_X, raw_y, color):
    """
    :param raw_X: training set X (n x 3) with (r,g,b)
    :param raw_y: class label of training set with (1,2,3) :{'red':1, 'green':2, 'blue':3}
    :return: training set X and  binary class label y (-1,1):{'not_blue':-1, 'blue':1}
    """
    assert raw_X.shape[0] == raw_y.shape[0]
    rgb = {'red': 1, 'green': 2, 'blue': 3}
    y: int
    new_y = raw_y[:]

    for i, y in enumerate(new_y):
        new_y[i] = 1 if y == rgb[color] else 0
    assert len(np.unique(raw_y)) == 2
    return raw_X, new_y


############################################################################
def main():
    # get_training data for red
    folder = './data/training'
    raw_X, raw_y = get_training_data(folder)
    # print(np.unique(raw_y))
    assert isinstance(raw_X, np.ndarray) and isinstance(raw_y, np.ndarray)

    X_red, y_red = binary_label(raw_X, raw_y, color='red')
    assert len(np.unique(raw_y)) == 2

    # get_training data for green
    raw_X, raw_y = get_training_data(folder)
    X_green, y_green = binary_label(raw_X, raw_y, color='green')
    assert len(np.unique(raw_y)) == 2

    # get_training data for blue
    raw_X, raw_y = get_training_data(folder)
    X_blue, y_blue = binary_label(raw_X, raw_y, color='blue')
    assert len(np.unique(raw_y)) == 2

    # print(np.unique(raw_y))
    # print(X_red.shape, y_red.shape)
    # print(np.unique(y_red))
    # print(X_green.shape, y_green.shape)
    # print(np.unique(y_green))
    # print(X_blue.shape, y_blue.shape)
    # print(np.unique(y_blue))
    # # 3 classes:{ r:1352 #g:1199 # b:1143 }
    # reds = sum(y_red == 1)
    # n_reds = sum(y_red == 0)
    # print(reds, n_reds)
    #
    # greens = sum(y_green == 1)
    # n_greens = sum(y_green == 0)
    # print(greens, n_greens)
    #
    # blues = sum(y_blue == 1)
    # n_blues = sum(y_blue == 0)
    # print(blues, n_blues)

    # set seed
    np.random.seed(0)

    ##################################################################################

    ''' Classifier_red '''
    classifier_red = LogisticRegression(color='red', pretrained=True)
    # start_time = time.time()
    # w_red = classifier_red.fit(X_red, y_red, learning_rate=1e-3, max_iter=10000 )
    # print("\n---Training RED Time %s seconds ---" % (time.time() - start_time))
    #
    y_pred_red = classifier_red.classify(X_red,)
    print("prediction:\n", y_pred_red)
    accuracy_red = sum(y_pred_red == y_red) / y_red.shape[0]
    print(accuracy_red)

    four_red = precision_recall_fscore_support(y_red, y_pred_red, average='binary')
    print(four_red)

    plt.plot(X_red @ classifier_red.w, 'r')
    plt.title('red')
    plt.show()
    #
    # classifier_red.save(color='red', fname='model.pkl')
    # classifier_red.load(color='red', fname='model.pkl')

    ############################################################################################################
    ''' Classifier_green '''
    classifier_green = LogisticRegression(color='green', pretrained=True)
    # start_time = time.time()
    # w_green = classifier_green.fit(X_green, y_green, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training GREEN Time %s seconds ---" % (time.time() - start_time))
    y_pred_green = classifier_green.classify(X_green)
    print("prediction:\n", y_pred_green)

    accuracy_green = sum(y_pred_green == y_green) / y_green.shape[0]
    print(accuracy_green)

    four_green = precision_recall_fscore_support(y_green, y_pred_green, average='binary')
    print(four_green)

    plt.plot(X_green @ classifier_green.w, 'g')
    plt.title('green')
    plt.show()
    #
    # classifier_green.save(color='green', fname='model.pkl')
    # classifier_green.load(color='green', fname='model.pkl')

    #################################################################################################################
    # Classifier_blue
    classifier_blue = LogisticRegression(color='blue', pretrained=True)
    # start_time = time.time()
    # w_blue = classifier_blue.fit(X_blue, y_blue, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training BLUE Time %s seconds ---" % (time.time() - start_time))
    y_pred_blue = classifier_blue.classify(X_blue)
    print("prediction:\n", y_pred_blue)

    accuracy_blue = sum(y_pred_blue == y_blue) / y_blue.shape[0]
    print(accuracy_blue)

    four_blue = precision_recall_fscore_support(y_blue, y_pred_blue, average='binary')
    print(four_blue)


    plt.plot(X_blue @ classifier_blue.w,label='X.T@w')
    #plt.plot(X_blue @ classifier_blue.w == 0,label='Decsion boundary', linewidth=7.0)
    #plt.legend()
    plt.title('blue')
    plt.show()
    #########################################################



    # classifier_blue.save(color='blue', fname='model.pkl')
    # classifier_blue.load(color='blue', fname='model.pkl')


if __name__ == "__main__":
    main()
    folder = './data/training'
    raw_X, raw_y = get_training_data(folder)
    # print(raw_X.shape)
    # classifier = PixelClassifier()
    # y_pred = classifier.classify(raw_X) + 1
    # accuracy = sum(y_pred == raw_y) / raw_y.shape[0]
    # print(accuracy)
    # # 0.9055793991416309
