# ==============================================================================
#
#       
#       Ahad Aghapour  Department of Computer Science Ozyegin University
#       aghapour.ah2@gmail.com
#
# ==============================================================================

'''

To run this code set the all parameters you want.
To evaluate performance of existing model embed in pickled file set justGetScoreFromPickle to True, otherwise 
if it set to False, it start to create train dataset and start to train the model which maybe time consuming with 
big dataset

'''

import pickle
import numpy as np
from captcha.image import ImageCaptcha
from scipy import misc
from scipy import signal
from matplotlib import pyplot as plt

from sklearn import ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


# this function calculate accuracy with max_digs lenght
# for example if your max_digs is 5, it convert all one digits to 5 digit and then calculate the accuracy
def scoreCalculate(testTarget, predictedLabels, max_digs):
    predictedLabels = predictedLabels.reshape(-1, max_digs)
    predictedLabels2 = np.zeros(predictedLabels.shape[0])
    for i in range(predictedLabels.shape[0]):
        predictedLabels2[i] = int(''.join(predictedLabels[i]))

    return np.mean(testTarget == predictedLabels2)


# split image nad padding that image in 60x60 pixels
def splitImage(bigImage, max_digs):
    images = []

    bigImage = bigImage.reshape(60, -1)

    vectorizedData = np.sum(bigImage, axis=0)

    imagesWidth = bigImage.shape[1]
    imageWidth = imagesWidth // max_digs

    imageSeparatorInteligent = np.ones(imagesWidth)

    # do the Left Trim at first
    widthThreshold = imageWidth // 2
    maxValue = np.max(vectorizedData[0:widthThreshold])
    bigestMaxValue = np.max(np.nonzero(vectorizedData[0:widthThreshold] == maxValue))
    imageSeparatorInteligent[0:bigestMaxValue] = 0

    # do the Right Trim
    maxValue = np.max(vectorizedData[imagesWidth - widthThreshold:imagesWidth])
    smallestMaxValue = np.min(np.nonzero(vectorizedData[imagesWidth - widthThreshold:imagesWidth] == maxValue))
    imageSeparatorInteligent[imagesWidth - widthThreshold + smallestMaxValue:imagesWidth] = 0

    # cut the Left and Right empty space
    deletedInx = np.where(imageSeparatorInteligent == 0)
    newbigImage = np.delete(bigImage, deletedInx, axis=1)

    # Split image again
    imagesWidth = newbigImage.shape[1]
    imageWidth = imagesWidth // max_digs
    padSize = 60 - imageWidth - 1

    # # plot the image and new image
    # plt.imshow(bigImage)
    # plt.show()
    # plt.imshow(newbigImage)
    # plt.show()

    # separate images
    for y in range(0, max_digs):
        imageSeg = newbigImage[:, imageWidth * y: imageWidth * (y + 1)]
        paddedImage = np.pad(imageSeg, [(0, 0), (0, padSize)], mode='constant', constant_values=255)
        # # plot the splited images
        # plt.imshow(imageSeg)
        # plt.show()
        # plt.imshow(paddedImage)
        # plt.show()
        images.append(paddedImage.flatten())

    return np.array(images)


# separate targets
def splitTarget(target):
    targets = []
    target = str(target)
    for i in range(len(target)):
        targetSeg = target[i]
        targets.append(targetSeg)
    return np.array(targets)

# split your data and target to 1 digit
def splitAllDataAndTargets(allImages, allTargets, max_digs):
    splitedData = []
    splitedTargets = []

    for i in range(len(allImages)):
        imgs = splitImage(allImages[i], max_digs=max_digs)
        if len(imgs) < max_digs:
            print(i, 'less index')
        for img in imgs:
            splitedData.append(img)

        targets = splitTarget(allTargets[i])
        for tar in targets:
            splitedTargets.append(tar)

    return np.array(splitedData), np.array(splitedTargets)


# generate data or test set, how many you want with how many digit
# for example you can set max_digs to 7 tor create 7 digits number like: 5647891
def generateData(n, max_digs):
    width = max_digs * 35
    imageCaptcha = ImageCaptcha()
    data = []
    target = []
    for i in range(n):
        x = np.random.randint(10 ** (max_digs - 1), 10 ** max_digs)
        img = misc.imread(imageCaptcha.generate(str(x)))
        img = np.mean(img, axis=2)[:, :width]
        thresholdBackgroundForeGround = np.mean(img.flatten()).astype(int)
        img[img < thresholdBackgroundForeGround] = 0
        img[img >= thresholdBackgroundForeGround] = 255
        img = signal.medfilt(img, (5, 5))
        data.append(img.flatten())
        target.append(x)
    return np.array(data), np.array(target)


# main section
def main(data_number, test_number, max_digs, pickledFile_name, justGetScoreFromPickle):
    print(__doc__)
    if justGetScoreFromPickle == False:
        # Generate Data , split Data, fit Data, save model to pickled file
        print('Star to generating captcha data (Train Mode) ....')
        imageData, target = generateData(n=data_number, max_digs=max_digs)
        # split Data and Targets
        print('Start to split %d digit captcha ....' % max_digs)
        splitedImageData, splitedTarget = splitAllDataAndTargets(imageData, target, max_digs=max_digs)
        # fit data
        print('Start to fit  with {0} data ....\n'.format(data_number))
        model1 = ensemble.RandomForestClassifier(min_samples_split=10)
        model1.fit(X=splitedImageData, y=splitedTarget)
        # Save model to pickle file
        print('Start to pickle your model')
        pickle._dump(model1, open(pickledFile_name, 'wb'))
    else:
        # Generate  test dataset, split test dataset, load the model from file, get score of that model
        print('Star to generating captcha data test (Test Mode) ....')
        testImageData, testTarget = generateData(n=test_number, max_digs=max_digs)
        # split Data and Targets
        print('Start to split %d digit captcha test ....' % max_digs)
        splitedTestImageData, splitedTestTarget = splitAllDataAndTargets(testImageData, testTarget, max_digs=max_digs)
        # load the model from pickle file
        print('Start to load pickle file and then evaluate our model...')
        loaded_model = pickle.load(open(pickledFile_name, 'rb'))
        # get score from pickle file model
        score = loaded_model.score(splitedTestImageData, splitedTestTarget)
        print('\nScore from pickled model for one number is:\t%.2f' % score)
        print('Score from pickled model for {} digit number is:\t{:.2f}'.format(max_digs,
            scoreCalculate(testTarget, loaded_model.predict(splitedTestImageData), max_digs)))
    print('\nfinished successfully!')


# Please set your all parameters from here
if __name__ == '__main__':
    # If you want just evaluate our model set it to Tru, if set to False it fit
    # your model that mybe time consuming and then evalute it
    justGetScoreFromPickle = True
    # number of digital you want to evaluate
    max_digs = 5
    # number of train dataset to fit with model
    data_number = 20000
    # number of test data to evalute our model from pickeld file
    test_number = 2000
    pickledFile_name = 'AhadGroup_prj4.pkl'

    # run the main function
    main(data_number, test_number, max_digs, pickledFile_name, justGetScoreFromPickle)
