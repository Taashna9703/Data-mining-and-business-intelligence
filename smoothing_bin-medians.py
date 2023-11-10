import numpy as np
import math
def createBins(lower_bound, width, quantity, dataset):
    bins = []
    for i in range(quantity):
        bins.append([])  # Append an empty list for each bin
        for j in range(i * width, i * width + width):
            if j < len(dataset):  # Make sure we don't go out of bounds
                bins[i].append(dataset[j])
    return bins
def binMedians(bins, quantity, width):
    for i in range(quantity):
        temp_bins = np.array(bins)
        median = np.median(temp_bins[i])
        for j in range(width):
            bins[i][j] = median
    return bins
dataset = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 5, 35, 35,
           36, 40, 45, 46, 52, 70]
b = len(dataset)
k = 3
w = int(b / k)
new_bins = createBins(0, k, w, dataset)
print("bins created")
print(new_bins)
bin_medians = binMedians(new_bins, w, k)
print("Smoothening by bin Medians")
print(bin_medians)
