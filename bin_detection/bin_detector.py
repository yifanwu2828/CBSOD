'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


import sys
import os
import pickle
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label, regionprops
from skimage import morphology
sys.path.append("../")



class BinDetector():
    def __init__(self):
        """
		Initilize your blue bin detector with the attributes you need,
		e.g., parameters of your classifier
		"""
        self.bin_blue = LogisticRegression(color='bin_blue', pretrained=True)
        self.bin_blue_w = self.bin_blue.w



    def segment_image(self, img):
        """
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64) / 255
        X = img.reshape((img.shape[0] * img.shape[1], 3))
        y = self.bin_blue.classify(X)
        mask_img = y.reshape((img.shape[0], img.shape[1])).astype('uint8')
        return mask_img

    def get_bounding_boxes(self, img):
        """
		Find the bounding boxes of the recycling bins
		Inputs:
		    img - original image
		Outputs:
			boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
			where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		"""

        # YOUR CODE HERE
        lst_bbox = []

        mask_img = img
        # erosion
        img_processed = morphology.erosion(mask_img)
        # img_processed = morphology.dilation(mask_img)
        #img_processed = morphology.closing(mask_img)
        # remove small object less than 1000 pixel
        img_processed = morphology.remove_small_objects(np.array(img_processed, dtype=bool), 1000, img_processed.ndim)

        #plt.imshow(img_processed)
        #plt.show()
        # label
        label_img = label(img_processed, connectivity=1)

        #max_size of image
        max_size = img.shape[0] * img.shape[1]
        for k, region in enumerate(regionprops(label_img)):
            (min_row, min_col, max_row, max_col) = region.bbox
            height = max_row - min_row
            width = max_col - min_col
            # two case consider, bin stand front and bin lay down
            shape_ratio = height / width

            #  ratio of num of detected region pixel vs num of entire pixel in image
            #  ratio of num of detected region pixel vs num of entire pixel in bbox
            #  shape_ratio h/w, two case consider, bin stand front and bin lay down 0.8
            if 0.01 * max_size < region.area and region.extent >= 0.45 and 1 <= shape_ratio <= 2:

                one_bbox_list = [min_col, min_row, max_col, max_row]
                lst_bbox.append(one_bbox_list)

        return lst_bbox


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
        if self.pretrained:
            self.load(color='blue', fname='model.pkl')
        else:
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

    def save(self, color, fname='model.pkl'):
        assert isinstance(color, str)
        fname = 'bin_detection/' +color + fname
        with open(fname, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, color, fname='model.pkl'):
        assert isinstance(color, str)
        # TODO: change this to below if submit
        fname = color + fname
        #fname = 'bin_detection/' + color + fname
        with open(fname, 'rb') as f:
            self.__dict__.update(pickle.load(f))


def load_mask2data():
    """
    load mask and convert masked pixels to training data and save with pickle for faster access
    :return: None
    """
    #load int8 data
    bin_blue_mask = dict()
    blue_mask = dict()
    black_mask = dict()
    red_mask = dict()
    yellow_mask = dict()
    green_mask = dict()

    bin_blue_uint8 = 'bin_blue_uint8.pkl'
    blue_uint8 = 'blue_uint8.pkl'
    black_uint8 = 'black_uint8.pkl'
    red_uint8 = 'red_uint8.pkl'
    yellow_uint8 = 'yellow_uint8.pkl'
    green_uint8 = 'green_uint8.pkl'

    with open(bin_blue_uint8, 'rb') as f0:
        bin_blue_mask = pickle.load(f0)
        print(len(bin_blue_mask))
    with open(blue_uint8, 'rb') as f1:
        blue_mask = pickle.load(f1)
        print(len(blue_mask))
    with open(black_uint8, 'rb') as f2:
        black_mask = pickle.load(f2)
        print(len(black_mask))
    with open(red_uint8, 'rb') as f3:
        red_mask = pickle.load(f3)
        print(len(red_mask))
    with open(yellow_uint8, 'rb') as f4:
        yellow_mask = pickle.load(f4)
        print(len(yellow_mask))
    with open(green_uint8, 'rb') as f5:
        green_mask = pickle.load(f5)
        print(len(green_mask))


    X0 = []
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []

    folder = './data/training'
    start_time = time.time()
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbmsk = bin_blue_mask[filename + '_0_Bin_blue']
        bmsk = blue_mask[filename+'_1_Blue']
        blcmsk = black_mask[filename+'_2_Black']
        rmsk = red_mask[filename+'_3_Red']
        ymsk = yellow_mask[filename+'_4_Yellow']
        gmsk = green_mask[filename+'_5_Green']

        for i in range(bbmsk.shape[0]):
            for j in range(bbmsk.shape[1]):
                if bbmsk[i, j] != 0:
                    X0.append(img[i, j, :].astype(np.float64)/255)

                if bmsk[i, j] != 0:
                    X1.append(img[i, j, :].astype(np.float64) / 255)

                if blcmsk[i, j] != 0:
                    X2.append(img[i, j, :].astype(np.float64) / 255)

                if rmsk[i, j] != 0:
                    X3.append(img[i, j, :].astype(np.float64) / 255)

                if ymsk[i, j] != 0:
                    X4.append(img[i, j, :].astype(np.float64) / 255)

                if gmsk[i, j] != 0:
                    X5.append(img[i, j, :].astype(np.float64) / 255)

    print("\n---Collect Data Time %s seconds ---" % (time.time() - start_time))

    # pickle save data -> list
    bin_blueX = 'bin_blueX.pkl'
    blueX = 'blueX.pkl'
    blackX = 'blackX.pkl'
    redX = 'redX.pkl'
    yellowX = 'yellowX.pkl'
    greenX = 'greenX.pkl'

    with open(bin_blueX, 'wb') as f0X:
        pickle.dump(X0, f0X)

    with open(blueX, 'wb') as f1X:
        pickle.dump(X1, f1X)

    with open(blackX, 'wb') as f2X:
        pickle.dump(X2, f2X)

    with open(redX, 'wb') as f3X:
        pickle.dump(X3, f3X)

    with open(yellowX, 'wb') as f4X:
        pickle.dump(X4, f4X)

    with open(greenX, 'wb') as f5X:
        pickle.dump(X5, f5X)


def relable_data(X0, X1, X2, X3, X4, X5, label):
    if label == 0:
        y0, y1, y2 = np.full(X0.shape[0], 1), np.full(X1.shape[0], 0), np.full(X2.shape[0], 0)
        y3, y4, y5 = np.full(X3.shape[0], 0), np.full(X4.shape[0], 0), np.full(X5.shape[0], 0)
        X, y = np.concatenate((X0, X1, X2, X3, X4, X5)), np.concatenate((y0, y1, y2, y3, y4, y5))
        print(label, ': X,y', X.shape, y.shape)
        assert 1 in y0
        assert 1 not in y5
        assert 1 not in y1
        assert 1 not in y2
        assert 1 not in y3
        assert 1 not in y4

    elif label == 1:
        y0, y1, y2 = np.full(X0.shape[0], 0), np.full(X1.shape[0], 1), np.full(X2.shape[0], 0)
        y3, y4, y5 = np.full(X3.shape[0], 0), np.full(X4.shape[0], 0), np.full(X5.shape[0], 0)
        X, y = np.concatenate((X0, X1, X2, X3, X4, X5)), np.concatenate((y0, y1, y2, y3, y4, y5))
        print(label, ': X,y', X.shape, y.shape)
        assert 1 in y1
        assert 1 not in y0
        assert 1 not in y5
        assert 1 not in y2
        assert 1 not in y3
        assert 1 not in y4

    elif label == 2:
        y0, y1, y2 = np.full(X0.shape[0], 0), np.full(X1.shape[0], 0), np.full(X2.shape[0], 1)
        y3, y4, y5 = np.full(X3.shape[0], 0), np.full(X4.shape[0], 0), np.full(X5.shape[0], 0)
        X, y = np.concatenate((X0, X1, X2, X3, X4, X5)), np.concatenate((y0, y1, y2, y3, y4, y5))
        print(label, ': X,y', X.shape, y.shape)
        assert 1 in y2
        assert 1 not in y0
        assert 1 not in y1
        assert 1 not in y5
        assert 1 not in y3
        assert 1 not in y4

    elif label == 3:
        y0, y1, y2 = np.full(X0.shape[0], 0), np.full(X1.shape[0], 0), np.full(X2.shape[0], 0)
        y3, y4, y5 = np.full(X3.shape[0], 1), np.full(X4.shape[0], 0), np.full(X5.shape[0], 0)
        X, y = np.concatenate((X0, X1, X2, X3, X4, X5)), np.concatenate((y0, y1, y2, y3, y4, y5))
        print(label, ': X,y', X.shape, y.shape)
        assert 1 in y3
        assert 1 not in y0
        assert 1 not in y1
        assert 1 not in y2
        assert 1 not in y5
        assert 1 not in y4
    elif label == 4:
        y0, y1, y2 = np.full(X0.shape[0], 0), np.full(X1.shape[0], 0), np.full(X2.shape[0], 0)
        y3, y4, y5 = np.full(X3.shape[0], 0), np.full(X4.shape[0], 1), np.full(X5.shape[0], 0)
        X, y = np.concatenate((X0, X1, X2, X3, X4, X5)), np.concatenate((y0, y1, y2, y3, y4, y5))
        print(label, ': X,y', X.shape, y.shape)
        assert 1 in y4
        assert 1 not in y0
        assert 1 not in y1
        assert 1 not in y2
        assert 1 not in y3
        assert 1 not in y5

    elif label == 5:
        y0, y1, y2 = np.full(X0.shape[0], 0), np.full(X1.shape[0], 0), np.full(X2.shape[0], 0)
        y3, y4, y5 = np.full(X3.shape[0], 0), np.full(X4.shape[0], 0), np.full(X5.shape[0], 1)
        X, y = np.concatenate((X0, X1, X2, X3, X4, X5)), np.concatenate((y0, y1, y2, y3, y4, y5))
        print(label, ': X,y', X.shape, y.shape)
        assert 1 in y5
        assert 1 not in y0
        assert 1 not in y1
        assert 1 not in y2
        assert 1 not in y3
        assert 1 not in y4
    assert X.shape[0] == y.shape[0]

    return X,y


def main():
    # load Training Data
    bin_blueX = 'bin_blueX.pkl'
    blueX = 'blueX.pkl'
    blackX = 'blackX.pkl'
    redX = 'redX.pkl'
    yellowX = 'yellowX.pkl'
    greenX = 'greenX.pkl'
    with open(bin_blueX, 'rb') as f0X:
        X0 = pickle.load(f0X)
    with open(blueX, 'rb') as f1X:
        X1 = pickle.load(f1X)

    with open(blackX, 'rb') as f2X:
        X2 = pickle.load(f2X)

    with open(redX, 'rb') as f3X:
        X3 = pickle.load(f3X)

    with open(yellowX, 'rb') as f4X:
        X4 = pickle.load(f4X)

    with open(greenX, 'rb') as f5X:
        X5 = pickle.load(f5X)

    X0, X1, X2 = np.vstack(X0), np.vstack(X1), np.vstack(X2)
    X3, X4, X5 = np.vstack(X3), np.vstack(X4), np.vstack(X5)
    print('Bin_Blue:', X0.shape, '\t', 'Blue:', X1.shape, '\t', 'Black:', X2.shape,)
    print('Red:', X3.shape, '\t', 'Yellow:', X4.shape, '\t', 'Green:', X5.shape, )

    X, y_bin_blue = relable_data(X0, X1, X2, X3, X4, X5, label=0)
    X, y_blue = relable_data(X0, X1, X2, X3, X4, X5, label=1)
    X, y_black = relable_data(X0, X1, X2, X3, X4, X5, label=2)
    X, y_red = relable_data(X0, X1, X2, X3, X4, X5, label=3)
    X, y_yellow = relable_data(X0, X1, X2, X3, X4, X5, label=4)
    X, y_green = relable_data(X0, X1, X2, X3, X4, X5, label=5)

    np.random.seed(1)

    # Classifier
    pretrained = True
    classifier_bin_blue = LogisticRegression(color='bin_blue', pretrained=pretrained)
    classifier_blue = LogisticRegression(color='blue', pretrained=pretrained)
    classifier_black = LogisticRegression(color='black', pretrained=pretrained)
    classifier_red = LogisticRegression(color='red', pretrained=pretrained)
    classifier_yellow = LogisticRegression(color='yellow', pretrained=pretrained)
    classifier_green = LogisticRegression(color='green', pretrained=pretrained)

    # Train
    # Bin_Blue
    # start_time = time.time()
    # w_bin_blue = classifier_bin_blue.fit(X, y_bin_blue, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training Bin_BLUE Time %s seconds ---" % (time.time() - start_time))
    y_pred_bin_blue = classifier_bin_blue.classify(X)
    print("prediction:\n", y_pred_bin_blue)
    accuracy_bin_blue = sum(y_pred_bin_blue == y_bin_blue) / y_bin_blue.shape[0]
    print(accuracy_bin_blue)
    plt.plot(X @ classifier_bin_blue.w, 'b')
    plt.title('Bin_Blue')
    plt.show()



    # Blue
    # start_time = time.time()
    # w_blue = classifier_blue.fit(X, y_blue, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training BLUE Time %s seconds ---" % (time.time() - start_time))
    y_pred_blue = classifier_blue.classify(X)
    print("prediction:\n", y_pred_blue)
    accuracy_blue = sum(y_pred_blue == y_blue) / y_blue.shape[0]
    print(accuracy_blue)
    plt.plot(X @ classifier_blue.w)
    plt.title('Blue')
    plt.show()

    # Black
    # start_time = time.time()
    # w_black = classifier_black.fit(X, y_black, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training BLACK Time %s seconds ---" % (time.time() - start_time))
    y_pred_black = classifier_black.classify(X)
    print("prediction:\n", y_pred_black)
    accuracy_black = sum(y_pred_black == y_black) / y_black.shape[0]
    print(accuracy_black)
    plt.plot(X @ classifier_black.w, 'k')
    plt.title('Black')
    plt.show()

    # Red
    # start_time = time.time()
    # w_red = classifier_red.fit(X, y_red, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training RED Time %s seconds ---" % (time.time() - start_time))
    y_pred_red = classifier_red.classify(X)
    print("prediction:\n", y_pred_red)
    accuracy_red = sum(y_pred_red == y_red) / y_red.shape[0]
    print(accuracy_red)
    plt.plot(X @ classifier_red.w, 'r')
    plt.title('Red')
    plt.show()

    # Yellow
    # start_time = time.time()
    # w_yellow = classifier_yellow.fit(X, y_yellow, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training YELLOW Time %s seconds ---" % (time.time() - start_time))
    y_pred_yellow = classifier_yellow.classify(X)
    print("prediction:\n", y_pred_yellow)
    accuracy_yellow = sum(y_pred_yellow == y_yellow) / y_yellow.shape[0]
    print(accuracy_yellow)
    plt.plot(X @ classifier_yellow.w, 'y')
    plt.title('Yellow')
    plt.show()

    # Green
    # start_time = time.time()
    # w_green = classifier_green.fit(X, y_green, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training GREEN Time %s seconds ---" % (time.time() - start_time))
    y_pred_green = classifier_green.classify(X)
    print("prediction:\n", y_pred_green)
    accuracy_green = sum(y_pred_red == y_green) / y_green.shape[0]
    print(accuracy_green)
    plt.plot(X @ classifier_green.w, 'g')
    plt.title('Green')
    plt.show()


    # classifier_bin_blue.save(color='bin_blue', fname='model.pkl')
    # classifier_blue.save(color='blue', fname='model.pkl')
    # classifier_black.save(color='black', fname='model.pkl')
    # classifier_red.save(color='red', fname='model.pkl')
    # classifier_yellow.save(color='yellow', fname='model.pkl')
    # classifier_green.save(color='green', fname='model.pkl')

    plt.plot(X @ classifier_bin_blue.w, 'b')
    plt.plot(X @ classifier_blue.w,)
    plt.plot(X @ classifier_black.w,)
    plt.plot(X @ classifier_red.w, 'r')
    plt.plot(X @ classifier_yellow.w, 'y')
    plt.plot(X @ classifier_green.w, 'g')
    plt.legend(('bin_blue', 'non_bin_blue', 'black', 'red', 'yellow', 'green'), loc='upper right')
    plt.title('All Color')
    plt.show()

    print(classifier_bin_blue.w,'\n')
    print(classifier_blue.w,'\n')
    print(classifier_black.w,'\n')
    print(classifier_red.w,'\n')
    print(classifier_yellow.w,'\n')
    print(classifier_green.w,'\n')

def show_mask():
    bin_detect = BinDetector()
    folder = './data/training'
    mask_img_dic = OrderedDict()
    img_dic = OrderedDict()

    start_time = time.time()
    for i, filename in enumerate(sorted(os.listdir(folder))):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_img_dic[filename] = bin_detect.segment_image(img)
        img_dic[filename] = img
        print(i + 1)
    print("\n---Generate Training mask_images Time %s seconds ---" % (time.time() - start_time))

    for key, value in mask_img_dic.items():
        plt.imshow(img_dic[key])
        plt.show()
        plt.imshow(value)
        plt.show()


if __name__ == '__main__':
    # load Training Data
    # bin_blueX = 'bin_blueX.pkl'
    # blueX = 'blueX.pkl'
    # blackX = 'blackX.pkl'
    # redX = 'redX.pkl'
    # yellowX = 'yellowX.pkl'
    # greenX = 'greenX.pkl'
    # with open(bin_blueX, 'rb') as f0X:
    #     X0 = pickle.load(f0X)
    # with open(blueX, 'rb') as f1X:
    #     X1 = pickle.load(f1X)
    #
    # with open(blackX, 'rb') as f2X:
    #     X2 = pickle.load(f2X)
    #
    # with open(redX, 'rb') as f3X:
    #     X3 = pickle.load(f3X)
    #
    # with open(yellowX, 'rb') as f4X:
    #     X4 = pickle.load(f4X)
    #
    # with open(greenX, 'rb') as f5X:
    #     X5 = pickle.load(f5X)
    #
    # X0, X1, X2 = np.vstack(X0), np.vstack(X1), np.vstack(X2)
    # X3, X4, X5 = np.vstack(X3), np.vstack(X4), np.vstack(X5)
    # print('Bin_Blue:', X0.shape, '\t', 'Blue:', X1.shape, '\t', 'Black:', X2.shape,)
    # print('Red:', X3.shape, '\t', 'Yellow:', X4.shape, '\t', 'Green:', X5.shape, )

    # # one vs all
    # # y0, y1, y2 = np.full(X0.shape[0], 0), np.full(X1.shape[0], 1), np.full(X2.shape[0], 2)
    # # y3, y4, y5 = np.full(X3.shape[0], 3), np.full(X4.shape[0], 4), np.full(X5.shape[0], 5)
    # # X, y = np.concatenate((X0, X1, X2, X3, X4, X5)), np.concatenate((y0, y1, y2, y3, y4, y5))
    # # assert len(np.unique(y)) == 6
    #
    # # Classifier
    # onevsallclassifier = OnevsAllLogisticRegression()
    #
    # y_pred_bin_ = onevsallclassifier.classify(X)
    # print("prediction:\n", y_pred_all)
    # accuracy_all = sum(y_pred_all == y) / y.shape[0]
    # print(accuracy_all)

    # # blue vs non blue
    # X, y_bin_blue = relable_data(X0, X1, X2, X3, X4, X5, label=0)
    #
    # np.random.seed(1)
    # pretrained = True
    # classifier_bin_blue = LogisticRegression(color='bin_blue', pretrained=pretrained)
    # # Train
    # # Bin_Blue
    # start_time = time.time()
    # w_bin_blue = classifier_bin_blue.fit(X, y_bin_blue, learning_rate=5e-2, max_iter=10000)
    # print("\n---Training Bin_BLUE Time %s seconds ---" % (time.time() - start_time))
    # y_pred_bin_blue = classifier_bin_blue.classify(X)
    # print("prediction:\n", y_pred_bin_blue)
    # accuracy_bin_blue = sum(y_pred_bin_blue == y_bin_blue) / y_bin_blue.shape[0]
    # print(accuracy_bin_blue)
    # plt.plot(X @ classifier_bin_blue.w, 'b')
    # plt.title('Bin_Blue')
    # plt.show()

    bin_detect = BinDetector()

    #folder = './data/training'
    folder = './data/validation'

    bin_detector = BinDetector()
    for i, filename in enumerate(sorted(os.listdir(folder))):

        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            img_mask = bin_detector.segment_image(img)
            plt.imshow(img_mask)
            plt.show()
            if filename == "0030.jpg":
                img_processed = morphology.erosion(img_mask)
                # img_processed = morphology.dilation(mask_img)
                # img_processed = morphology.closing(mask_img)
                # remove small object less than 1000 pixel
                # img_processed = morphology.remove_small_objects(np.array(img_processed, dtype=bool), 1000,
                #                                                 img_processed.ndim)
            # plt.imshow(img_mask)
            # plt.show()
            bbox_list = bin_detector.get_bounding_boxes(img_mask)
            print(bbox_list)

            img_result = img
            cv2.imshow('Origin' + filename, img_result)

            k = cv2.waitKey(2000) & 0xFF
            if k == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
            for box in bbox_list:
                min_col, min_row, max_col, max_row = box[0], box[1], box[2], box[3]
                cv2.rectangle(img_result, (min_col, min_row), (max_col, max_row), (0, 0, 255), 2)
                img_out=cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                #plt.imshow(img_out)
                #plt.show()
                cv2.imshow('Result' + filename, img_result)
                k = cv2.waitKey(500) & 0xFF
                if k == 27:  # wait for ESC key to exit
                    cv2.destroyAllWindows()






