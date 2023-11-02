#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import threading
import time
import datetime
from greenness_track_toolkit.agent.core.colletor.collector_base import Collector
import psutil

from greenness_track_toolkit.agent.models.energy import Energy
from greenness_track_toolkit.utils import get_logger

NVIDIA_SMI_LIST = "/usr/bin/nvidia-smi --list-gpus"
NVIDIA_SMI_LIST_OUT = 3  # --list-gpus
NVIDIA_SMI_QUERY = "/usr/bin/nvidia-smi --loop-ms=500 --query-gpu=gpu_uuid,timestamp,power.draw --format=csv,noheader,nounits"
NVIDIA_SMI_QUERY_OUT = 3  # --query-gpu
NVIDIA_SMI_QUERY_APPS = "/usr/bin/nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader"
NVIDIA_SMI_QUERY_APPS_OUT = 2  # --query-compute-apps


class GPU_Dev:
    def __init__(self):

        self._lock = threading.Lock()
        self.dt = None
        self.energy = 0.0


    def update(self, dt, power):
        self._lock.acquire()
        if self.dt is None:

            self.dt = dt
            self.energy = 0.0
        else:
            if dt > self.dt:
                delta_time = (dt - self.dt).total_seconds()
                self.energy += (power * delta_time)
                self.dt = dt
            else:
                get_logger().warning("old datetime %s is greater than new datetime %s\n" %
                                    (self.dt, dt))
        _e = self.energy
        self._lock.release()
        return _e


    def reset(self):
        self._lock.acquire()
        _e = self.energy
        _dt = self.dt
        self.energy = 0.0
        self._lock.release()
        return _e, _dt

class GPU:
    def __init__(self, debug=False):
        self.gpu = {}
        self._debug = debug
        # format: GPU 0: Tesla V100-SXM2-16GB (UUID: GPU-def56341-99b3-f22a-f3a8-2bc96ac692a3)
        proc = subprocess.Popen(NVIDIA_SMI_LIST, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        try:
            outs, errs = proc.communicate(timeout=3)
            if errs:
                print(errs)
                raise subprocess.CalledProcessError
            lines = outs.splitlines()
            for ln in lines:
                parts = ln.strip().decode().split(':')
                if len(parts) != NVIDIA_SMI_LIST_OUT:
                    continue
                uuid = parts[2].strip().strip(")")
                self.gpu[uuid] = GPU_Dev()

        except subprocess.TimeoutExpired as ex:
            proc.kill()
            raise ex

    def update(self, uuid, dt, power):
        if uuid not in self.gpu:
            get_logger().warning("the uuid %s is incorrect\n" % uuid)
            return
        gpu_dev = self.gpu[uuid]
        energy = gpu_dev.update(dt, power)

        if self._debug:
            get_logger().info("update gpu %s: datetime %s energy %f" % (uuid, dt, energy,))


    def get(self, uuid):
        if uuid not in self.gpu:
            get_logger().warning("the uuid %s is incorrect\n" % uuid)
            return 0.0
        gpu_dev = self.gpu[uuid]
        energy, dt = gpu_dev.reset()

        if self._debug:
            get_logger().info("reset gpu %s: datetime %s energy %f" % (uuid, dt, energy,))

        return energy


def parse_query_gpu(gpu, line):
    # format: GPU-997a7dbd-329e-c92b-6a55-f5842b9e5421, 2022/11/21 15:29:33.123, 256.32
    parts = line.strip().decode().split(',')
    if len(parts) != NVIDIA_SMI_QUERY_OUT:
        return
    uuid = parts[0].strip()
    time_str = parts[1].strip()
    dt = datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S.%f')
    power = float(parts[2].strip())
    gpu.update(uuid, dt, power)


def gpu_collect(p, gpu):
    assert isinstance(p, subprocess.Popen)
    while p.poll() is None:
        # format: GPU-c12dd5e2-f6d8-d6fa-b85d-86c75ccdcc0d, 2022/11/22 11:24:09.955, 43.38
        line = p.stdout.readline().strip()
        parse_query_gpu(gpu, line)


class NvidiaCollector(Collector):
    def __init__(self):
        super().__init__()
        self._pid = psutil.Process().pid
        self._gpu_uuid = []
        self._gpu = None
        self._proc = None
        self._th = None

    def check(self):
        if len(self._gpu_uuid) == 0:
            # format: GPU-9255e099-aa1c-c951-b642-6ec3202884a8, 175520
            proc = subprocess.Popen(NVIDIA_SMI_QUERY_APPS, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            try:
                outs, errs = proc.communicate(timeout=3)
                if errs:
                    print(errs)
                    raise subprocess.CalledProcessError
                lines = outs.splitlines()
                for ln in lines:
                    parts = ln.strip().decode().split(',')
                    if len(parts) != NVIDIA_SMI_QUERY_APPS_OUT:
                        continue
                    uuid = parts[0].strip()
                    pid = int(parts[1].strip())
                    if pid == self._pid:
                        self._gpu_uuid.append(uuid)
            except subprocess.TimeoutExpired as ex:
                proc.kill()
                raise ex
        if len(self._gpu_uuid) == 0:
            get_logger().warning("no gpu is assigned to the process %d\n" % self._pid)
            return False
        return True

    def start(self):
        if not self.check():
            return

        if self._gpu is None:
            self._gpu = GPU()

            proc = subprocess.Popen(NVIDIA_SMI_QUERY, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            t = threading.Thread(target=gpu_collect, args=[proc, self._gpu])
            t.start()
            self._proc = proc
            self._th = t

    def close(self):
        if self._gpu is not None:
            self._proc.terminate()
            time.sleep(1)
            self._th.join()

    def delta(self, duration):
        if len(self._gpu_uuid) == 0:
            self.start()
            return Energy(energy=0.0)
        energy = 0.0
        for uuid in self._gpu_uuid:
            energy += self._gpu.get(uuid)
        return Energy(energy)