from datetime import datetime


def showExecutionTime(startTime):
    timeElapsed = datetime.now() - startTime
    print('\n Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
    print("pc_czarny \t Time elapsed (hh:mm:ss.ms) 0:02:05.02146")
    print("laptop \t Time elapsed (hh:mm:ss.ms) 0:02:56.188259")
    print("bestia \t Time elapsed (hh:mm:ss.ms) 0:01:16.143156")
