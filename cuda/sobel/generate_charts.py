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
        for thread in np.sort(np.unique(X)):
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
    # plt.bar(range(len(Y)), Y)
    # plt.xticks(range(len(X)), X)
    plt.plot(X, Y, '-go')
    plt.xlabel("Block Size (px)")
    plt.ylabel(ylabel)
    plt.show()
    fig.savefig(outFile)


def processCCCStats():
    # Read in csv of copy compute copy stats
    xory = []
    blocksize = []
    cccTimes = []
    with open(copyComputeCopyFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            xory.append(row[0])
            blocksize.append(int(row[1]))
            cccTimes.append(float(row[2]))

    # Send to avergae ccctimes by filter size
    blockSizeX = []
    blockSizeY = []
    cccTimesX = []
    cccTimesY = []
    for i, item in enumerate(xory):
        if (item == "x"):
            blockSizeX.append(blocksize[i])
            cccTimesX.append(cccTimes[i])
        else:
            blockSizeY.append(blocksize[i])
            cccTimesY.append(cccTimes[i])


    cccTimesX = compress_data(blockSizeX, cccTimesX)
    cccTimesY = compress_data(blockSizeY, cccTimesY)


    print(cccTimes)
    # Plot times
    plot_data(np.sort(np.unique(blockSizeX)), cccTimesX, "Copy-Compute-Copy Time for X vs Block Size", "Copy-Compute-Copy Time for X (s)", "data/ccc_stats_x.pdf")
    plot_data(np.sort(np.unique(blockSizeX)), np.log(cccTimesX), "ln( Copy-Compute-Copy Time for X ) vs Block Size", "ln( Copy-Compute-Copy Time for X) (s)", "data/ccc_stats_ln_x.pdf")
    # Plot times
    plot_data(np.sort(np.unique(blockSizeY)), cccTimesY, "Copy-Compute-Copy Time for Y vs Block Size", "Copy-Compute-Copy Time for Y (s)", "data/ccc_stats_y.pdf")
    plot_data(np.sort(np.unique(blockSizeY)), np.log(cccTimesY), "ln( Copy-Compute-Copy Time for Y ) vs Block Size", "ln( Copy-Compute-Copy Time for Y) (s)", "data/ccc_stats_ln_y.pdf")

    # Plot X vs Y times
    fig = plt.figure()
    fig.suptitle("X vs Y Compute times")
    plt.plot(np.sort(np.unique(blockSizeX)), cccTimesX, '-ro')
    plt.plot(np.sort(np.unique(blockSizeY)), cccTimesY, '-go')    
    plt.xlabel("Block Size (px)")
    plt.ylabel("Compute time for X(red) & Y(green)")
    plt.show()
    fig.savefig("data/ccc_stats_xy.pdf")


    gsTimes = []
    with open(goldenStandardTimingFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            gsTimes.append(float(row[0]))

    cpuTimes = np.average(gsTimes)
    cpuTimesRepeat = np.repeat(cpuTimes, 7)
    print(cpuTimes)

    # Plot cpu vs gpu times
    fig = plt.figure()
    fig.suptitle("X vs Y vs CPU Compute times")
    plt.plot(np.sort(np.unique(blockSizeX)), cccTimesX, '-bo')
    plt.plot(np.sort(np.unique(blockSizeY)), cccTimesY, '-go') 
    plt.plot(np.sort(np.unique(blocksize)),  cpuTimesRepeat, '-r')   
    plt.xlabel("Block Size (px)")
    plt.ylabel("Compute time for X(blue), Y(green), CPU(red)")
    plt.show()
    fig.savefig("data/ccc_stats_xy_cpu.pdf")

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
    # compareCPUvsGPU()

if __name__ == '__main__':
    main()