#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import defaultdict
from greenness_track_toolkit.agent.utils.exception import NotSupportedCPU

from greenness_track_toolkit.utils import get_logger


class CPUBrand:
    INTEL = 1
    AMD = 2


def read_uint(filename):
    v = -1
    with open(filename, mode='r') as f:
        v = int(f.read())
    return v


def get_package_id(cpu):
    return read_uint("/sys/devices/system/cpu/cpu%d/topology/physical_package_id" % cpu)


def parse_cpus(cpus):
    cpus_list = []
    for cpuset in cpus.split(","):
        if cpuset == "":
            continue

        if "-" in cpuset:
            ranges = cpuset.split("-")
            if len(ranges) != 2:
                get_logger().error("invalid cpu range: %s", cpuset)
                return []
            start = int(ranges[0])
            end = int(ranges[1])
            for cpu in range(start, end + 1):
                cpus_list.append(cpu)
        else:
            if cpuset.isdigit():
                cpu = int(cpuset)
                cpus_list.append(cpu)
            else:
                get_logger().error("invalid cpu  %s", cpuset)
                return []

    return cpus_list


def all_present_cpus():
    fn = "/sys/devices/system/cpu/present"
    with open(fn, mode='r') as f:
        cpus = parse_cpus(f.read().strip())

    if not len(cpus):
        raise Exception("failed to get present cpu")
    return cpus


class CPU:
    def __init__(self):
        present_cpus = all_present_cpus()
        cpu2pkg = {}
        pkg2cpu = defaultdict(list)
        for cpu in present_cpus:
            pkg_id = get_package_id(cpu)
            if pkg_id < 0:
                raise Exception("invalid cpu id" % cpu)
            cpu2pkg[cpu] = pkg_id
            pkg2cpu[pkg_id].append(cpu)

        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                n = line.split()
                if len(n) < 3:
                    continue
                if n[0] == "vendor_id":
                    self.vendor = n[2]
                elif n[0] == "cpu" and n[1] == "family":
                    self.family = int(n[3])
                elif n[0] == "model" and n[1] == ":":
                    self.model = int(n[2])

        if self.vendor is None or \
                self.family is None or \
                self.model is None:
            raise Exception("can't get cpuinfo correctly")

        self.cpu2pkg = cpu2pkg
        self.pkg2cpu = pkg2cpu
        if self.vendor.lower().find("amd") > 0:
            self.brand = CPUBrand.AMD
        elif self.vendor.lower().find("intel") > 0:
            self.brand = CPUBrand.INTEL
        else:
            raise NotSupportedCPU()

    def get_lead_cpuid(self, pkg_id):
        cpus = self.pkg2cpu[pkg_id]
        if len(cpus) <= 0:
            get_logger().error("package %d not found" % pkg_id)
            return 0
        return cpus[0]

    def get_pkgs(self):
        pkgs = []
        for pkg_id in self.pkg2cpu:
            if pkg_id >= 0:
                pkgs.append(pkg_id)
        return pkgs