from datetime import datetime

from MnistOfficial import start_time


def timer():
    time_elapsed = datetime.now() - start_time
    print('\n Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print("pc_czarny \t Time elapsed (hh:mm:ss.ms) 0:02:05.02146")
    print("laptop \t Time elapsed (hh:mm:ss.ms) 0:02:56.188259")
    print("bestia \t Time elapsed (hh:mm:ss.ms) 0:01:16.143156")