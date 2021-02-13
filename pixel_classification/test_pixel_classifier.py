'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division

from generate_rgb_data import read_pixels
from pixel_classifier_copy import PixelClassifier

import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from pixel_classifier_copy import *


def main():
  # test the classifier red

  folder = 'data/validation/green'

  X1 = read_pixels(folder)
  y1 = np.full(X1.shape[0], 1)
  myPixelClassifier = LogisticRegression(color='green', pretrained=True)

  y_pred_r = myPixelClassifier.classify(X1)

  print('Precision: %f' % (sum(y_pred_r == 1) / y1.shape[0]))

  acc = accuracy_score(y1, y_pred_r)
  print(acc)
  four = precision_recall_fscore_support(y1, y_pred_r,)
  print(four)

  # #test the classifier green
  # folder = 'data/validation/green'
  #
  # X2 = read_pixels(folder)
  # y2 = np.full(X2.shape[0], 1)
  # myPixelClassifier = LogisticRegression(color='green', pretrained=True)
  #
  # y_pred_g = myPixelClassifier.classify(X1)
  #
  # print('Precision: %f' % (sum(y_pred_g == 1) / y2.shape[0]))
  #
  # acc = accuracy_score(y2, y_pred_r)
  # print(acc)
  # four = precision_recall_fscore_support(y2, y_pred_r, average='binary')
  # print(four)
  #
  # # #test the classifier blue
  # # folder = 'data/validation/blue'
  # #
  # # X = read_pixels(folder)
  # # myPixelClassifier = PixelClassifier(color='blue', pretrained=True)
  # #
  # # y = myPixelClassifier.classify(X)
  # #
  # # print('Precision: %f' % (sum(y == 1) / y.shape[0]))


if __name__ == '__main__':
  main()
  print('\n'*5)

  folder = './data/validation'
  raw_X, raw_y = get_training_data(folder)
  classifier = PixelClassifier()
  y_pred = classifier.classify(raw_X) + 1
  accuracy = sum(y_pred == raw_y) / raw_y.shape[0]
  print(accuracy)

  acc = accuracy_score(raw_y, y_pred)
  print(acc)
  four = precision_recall_fscore_support(raw_y, y_pred, average='weighted')
  print(four,'\n')

  print('red:',classifier.red_w, '\n')
  print('green', classifier.green_w, '\n')
  print('blue',classifier.blue_w, '\n')