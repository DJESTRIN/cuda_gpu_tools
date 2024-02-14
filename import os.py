import os
import sys
import json
import time
import argparse
import xmltodict
import subprocess as sp
from collections import namedtuple
#from torch.utils.tensorboard import SummaryWriter

def main():
    # Get Nvidia SMI log message
    log, _ = sp.Popen(['nvidia-smi', '-q', '-x'], stdout=sp.PIPE).communicate()
    log = log.decode('UTF-8')
    log = xmltodict.parse(log)['nvidia_smi_log']

    # Parse information
    driver_version = log['driver_version']
    gpu_name = log['gpu']['product_name']
    fan_speed = proc(log['gpu']['fan_speed'])
    cuda_version = log['cuda_version']
    used_memory = proc(log['gpu']['fb_memory_usage']['used'])
    total_memory = proc(log['gpu']['fb_memory_usage']['total'])
    memory_usage = used_memory / total_memory * 100
    gpu_usage = proc(log['gpu']['utilization']['gpu_util'])
    gpu_temperature = proc(log['gpu']['temperature']['gpu_temp'])
    print(gpu_usage)
    return
   

def parse_proc(p):
    name = os.path.basename(p['process_name'])
    mem = proc(p['used_memory'])
    pid = p['pid']

    return Process(name, pid, mem)

if __name__ == "__main__":
    stop_time = 5 # seconds
    start_time=time.time()
    pullcuda=True
    while pullcuda:
        main()
        time.sleep(0.01)
        current_time = time.time()
        if (current_time-start_time) > stop_time:
            pullcuda=False

