import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

accuracyFile = "data/accuracy_stats.csv"
loadTimeFile = "data/lt_stats.csv"
copyComputeCopyFile = "data/ccc_stats.csv"
goldenStandardTimingFile = "data/gs_stats.csv"

def compress_data(X, Y, compress_type="average"):
    data = {}
    for i in range(len(X)):
        try:
            data[X[i]].append(Y[i])
        except KeyError:
            data[X[i]] = [Y[i]]
    newY = []
    if(compress_type == "average"):
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
    plt.xlabel("Filter Size (px)")
    plt.ylabel(ylabel)
    plt.show()
    fig.savefig(outFile)


def processCCCStats():
    # Read in csv of copy compute copy stats
    filtersize = []
    cccTimes = []
    with open(copyComputeCopyFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            filtersize.append(int(row[0]))
            cccTimes.append(float(row[1]))

    # Send to avergae ccctimes by filter size
    cccTimes = compress_data(filtersize, cccTimes)

    # Plot times
    plot_data(np.unique(filtersize), cccTimes, "Copy-Compute-Copy Time vs Filter Size", "Copy-Compute-Copy Time (s)", "data/ccc_stats.pdf")
    plot_data(np.unique(filtersize), np.log(cccTimes), "ln( Copy-Compute-Copy Time ) vs Filter Size", "ln( Copy-Compute-Copy Time ) (s)", "data/ccc_stats_ln.pdf")

def getAverageLoadTime():
    loadTimes = []
    with open(loadTimeFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            loadTimes.append(float(row[0]))

    average = np.average(loadTimes)
    return average

def compareCPUvsGPU():
    # load GPU (ccc) times vs CPU times (gs)
    # Read in csv of copy compute copy stats
    filtersize = []
    cccTimes = []
    with open(copyComputeCopyFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            filtersize.append(int(row[0]))
            cccTimes.append(float(row[1]))
    # Read in csv of copy compute copy stats
    gsTimes = []
    with open(goldenStandardTimingFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            gsTimes.append(float(row[1]))

    cccTimes = compress_data(filtersize, cccTimes)
    gsTimes = compress_data(filtersize, gsTimes)
    filtersize = np.unique(filtersize)

    fig = plt.figure()
    fig.suptitle("CPU vs GPU Processing Speedup")
    plt.plot(filtersize, cccTimes, '-go')
    plt.plot(filtersize, gsTimes, '-ro')
    plt.xlabel("Filter Size (px)")
    plt.ylabel("CPU(red) and GPU(green) times (s)")
    plt.show()
    fig.savefig("data/cpu_gpu.pdf")

    fig = plt.figure()
    fig.suptitle("ln( CPU vs GPU Processing Speedup )")
    plt.plot(filtersize, np.log(cccTimes), '-go')
    plt.plot(filtersize, np.log(gsTimes), '-ro')
    plt.xlabel("Filter Size (px)")
    plt.ylabel("ln( CPU(red) and GPU(green) times ) (s)")
    plt.show()
    fig.savefig("data/cpu_gpu_ln.pdf")


def  main():
    processCCCStats()
    loadTimeAverage = getAverageLoadTime()
    print(loadTimeAverage)
    compareCPUvsGPU()

if __name__ == '__main__':
    main()