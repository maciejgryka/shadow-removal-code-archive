import numpy as np
import matplotlib.pyplot as plt

features = np.genfromtxt('features_test.csv', delimiter=',')
labels_gt = np.genfromtxt('labels_test_gt.csv', delimiter=',')
labels = np.genfromtxt('labels_test.csv', delimiter=',')
plt.plot(features, labels_gt[:,0], 'b', label='1st label dimension')
plt.plot(features, labels_gt[:,1], 'g', label='2nd label dimension')
plt.plot(features, labels[:,0], '.b', label='1st dimension prediction')
plt.plot(features, labels[:,1], '.g', label='2nd dimension prediction')
plt.legend()
plt.show()