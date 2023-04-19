import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def read_histograms_from_csv(path):
    # read in the training data
    angle_histograms = []
    labels = []
    with open(path, 'r') as csv_file:
        for line in csv_file:
            line = line.strip()
            if line == '':
                continue
            line = line.split(',')
            angle_histogram = [float(x) for x in line[:-1]]
            label = line[-1]
            angle_histograms.append(angle_histogram)
            labels.append(label)
    return angle_histograms, labels


def train_svm_model(angle_histograms, labels):
    svm = SVC()
    svm.fit(angle_histograms, labels)
    return svm


def save_svm_model(svm, path):
    with open(path, 'wb') as model_file:
        pickle.dump(svm, model_file)


def load_svm_model(path):
    with open(path, 'rb') as model_file:
        svm = pickle.load(model_file)
    return svm


def train_and_save_model():
    angle_hist, labels = read_histograms_from_csv('training_data.csv')

    print(angle_hist)

    angle_hist = np.array(angle_hist)
    labels = np.array(labels)

    angle_hist_train, angle_hist_test, labels_train, labels_test = train_test_split(angle_hist, labels, test_size=0.2)

    model = train_svm_model(angle_hist_train, labels_train)

    accuracy = model.score(angle_hist_test, labels_test)
    print(f'Accuracy: {accuracy}')

    save_svm_model(model, 'model.pkl')
