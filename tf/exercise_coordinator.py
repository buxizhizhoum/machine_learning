#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import threading
import time


# def loop(coord, worker_id):
#     while not coord.should_stop():
#         if np.random.rand() < 0.1:
#             print("stop %s" % worker_id)
#             coord.request_stop()
#         else:
#             print("working %s" % worker_id)
#
#         time.sleep(1)


coord = tf.train.Coordinator()

# threads = [threading.Thread(target=loop, args=(coord, i,)) for i in range(5)]
#
# for i in threads:
#     i.start()
#
# coord.join()


class Threads(threading.Thread):
    def __init__(self, worker_id):
        super(Threads, self).__init__()
        self.worker_id = worker_id

    def run(self):
        while not coord.should_stop():
            if np.random.rand() < 0.1:
                print("stop %s" % self.worker_id)
                coord.request_stop()
            else:
                print("working %s" % self.worker_id)

            time.sleep(1)


threads = [Threads(i) for i in range(5)]
for i in threads:
    i.start()




