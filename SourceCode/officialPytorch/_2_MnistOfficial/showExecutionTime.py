from datetime import datetime


def showExecutionTime(startTime):
    timeElapsed = datetime.now() - startTime
    print('\n Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))
