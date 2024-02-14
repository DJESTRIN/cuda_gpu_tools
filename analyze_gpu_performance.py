import os
import sys
import json
import time
import argparse
import xmltodict
import subprocess as sp
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter

timestr = time.strftime("%Y%m%d%H%M%S")
output_folder = '/home/fs01/dje4001/cuda_gpu_tools/run'+timestr+'/'
os.mkdirs(output_folder)
writer = SummaryWriter([output_folder])


def main(time_oh):
    # Get Nvidia SMI log message
    log, _ = sp.Popen(['nvidia-smi', '-q', '-x'], stdout=sp.PIPE).communicate()
    log = log.decode('UTF-8')
    log = xmltodict.parse(log)['nvidia_smi_log']
    gpu_usage = proc(log['gpu']['utilization']['gpu_util'])
    writer.add_scalar("Loss/train", gpu_usage, time_oh)
    return 

def proc(x):
    return float(x.split(' ')[0])

if __name__ == "__main__":
    stop_time = 5 # seconds
    start_time=time.time()
    pullcuda=True
    while pullcuda:
        time.sleep(0.01)
        current_time = time.time()
        main(current_time)
        if (current_time-start_time) > stop_time:
            pullcuda=False

