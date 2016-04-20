import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


def compress_data(X, Y):
    data = {}

    for i in range(len(X)):
        try:
            data[X[i]].append(Y[i])
        except KeyError:
            data[X[i]] = [Y[i]]

    newY = []
    for thread in np.unique(X):
        sum = 0
        for value in data[thread]:
            sum  = sum + value
        sum = sum / len(data[thread])
        data[thread] = sum

    for thread in np.unique(X):
        newY.append(data[thread])

    return newY


def plot_data(X, Y, title, ylabel, outFile):
    fig = plt.figure()
    fig.suptitle(title)
    plt.plot(X, Y, '-go')
    plt.xlabel("Number of Threads")
    plt.ylabel(ylabel)
    plt.show()
    fig.savefig(outFile)



def main(argv):
    # Open csv, put data into column vectors
    numThreads = []
    wallClockTimes = []
    perVectorWallClockTimes = []
    with open(argv[1], 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            numThreads.append(int(row[0]))
            wallClockTimes.append(float(row[1]))
            perVectorWallClockTimes.append(float(row[4]))
    
    # Compress into averages for each number of threads
    wallClockTimes = compress_data(numThreads, wallClockTimes)
    perVectorWallClockTimes = compress_data(numThreads, perVectorWallClockTimes)
    numThreads = np.unique(numThreads)
    # plot
    plot_data(np.unique(numThreads), wallClockTimes, "Wall Clock Times vs Number of Threads", "Wall Clock Times(s)", argv[2] + "_wall_clock.pdf")
    plot_data(np.unique(numThreads), perVectorWallClockTimes, "per Vector Wall Clock Times vs Number of Threads", "per Vector Wall Clock Times (s)", argv[2]+"_perVector.pdf")
    
    # Write compressed to excel
    with open(argv[2]+".csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        for i, elem in enumerate(numThreads):
            writer.writerow([numThreads[i], wallClockTimes[i], perVectorWallClockTimes[i]])


if __name__ == '__main__':
    main(sys.argv)
