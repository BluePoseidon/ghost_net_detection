import random
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


NUM_BINS = 8
NUM_POSITIVE_SAMPLES = 1000
NUM_NEGATIVE_SAMPLES = 1000


def create_positive_histogram(num_samples):
    sigma = pi/12

    mu1 = random.uniform(0, pi)

    mu2 = mu1 + pi/2
    mu2 = random.uniform(mu2 - pi/4, mu2 + pi/4)

    z = np.random.binomial(1, 0.5, num_samples)
    x = np.zeros(num_samples)
    x[z == 0] = np.random.normal(mu1, sigma, np.sum(z == 0))
    x[z == 1] = np.random.normal(mu2, sigma, np.sum(z == 1))
    x = np.mod(x, pi)

    hist, bin_edges = np.histogram(x, bins=NUM_BINS, range=(0, pi))

    return hist, bin_edges


def create_negative_histogram(num_samples):
    x = np.random.uniform(0, pi, num_samples)
    hist, bin_edges = np.histogram(x, bins=NUM_BINS, range=(0, pi))
    return hist, bin_edges


def save_histograms_to_csv(histograms, path):
    histograms = histograms.astype(int)
    np.savetxt(path, histograms, delimiter=',', fmt='%i')

def create_synthetic_histogram_data():
    negative_histograms = []
    for i in range(NUM_NEGATIVE_SAMPLES):
        num_lines = random.randint(0, 40)
        hist, bin_edges = create_negative_histogram(num_lines)
        negative_histograms.append(hist)

    positive_histograms = []
    for i in range(NUM_POSITIVE_SAMPLES):
        num_lines = random.randint(20, 100)
        hist, bin_edges = create_positive_histogram(num_lines)
        positive_histograms.append(hist)

    # append class labels
    negative_histograms = np.array([np.append(hist, 0) for hist in negative_histograms])
    positive_histograms = np.array([np.append(hist, 1) for hist in positive_histograms])

    histograms = np.concatenate((negative_histograms, positive_histograms))
    np.random.shuffle(histograms)

    print(f'{histograms = }')

    save_histograms_to_csv(histograms, 'training_data.csv')
