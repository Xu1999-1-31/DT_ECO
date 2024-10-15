import DataBuilder
from multiprocessing import Process
import threading
import time
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var

cores = 6

def BuildAllData(design):
    DataBuilder.BuildCellLayout(design)
    DataBuilder.BuildMetalLayer(design)
    for scale in Global_var.scales:
        DataBuilder.BuildRouteCongestion(design, scale)
        DataBuilder.BuildCellDensity(design, scale)
    DataBuilder.BuildDrcMap(design)
    DataBuilder.BuildVerilog(design)
    DataBuilder.BuildPtRpt(design)
    DataBuilder.BuildTimingArc(design)
    DataBuilder.BuildEndPoint(design)
    DataBuilder.BuildGlobalTimingData(design)
    DataBuilder.BuildPortData(design)
    DataBuilder.BuildPinLayout(design)
    return

def MultiCoreAccelerator():
    all_done = threading.Event()
    def output_status():
        dot_count = 0
        while not all_done.is_set():
            dot_count = dot_count + 1 if dot_count < 6 else 0
            if(dot_count == 0):
                print(f"\rBuilding data with {cores} cores " + " " * 6, end="")
            else:
                print(f"\rBuilding data with {cores} cores " + "." * dot_count, end="")
                time.sleep(1)
        print()

    status_thread = threading.Thread(target=output_status)
    status_thread.start()

    try:
        max_processes = cores  # max process number
        processes = []
        for design in Global_var.Designs:
            if len(processes) >= max_processes:
                processes[0].join()  # wait for the earliest process
                processes.pop(0)  # remove complete process

            process = Process(target=BuildAllData, args=(design,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()  # wait for all process complete

        all_done.set()
    except Exception as e:
        print(f"Error occurred: {e}")
        all_done.set()
        raise
    status_thread.join()
    return

# MultiCoreAccelerator()