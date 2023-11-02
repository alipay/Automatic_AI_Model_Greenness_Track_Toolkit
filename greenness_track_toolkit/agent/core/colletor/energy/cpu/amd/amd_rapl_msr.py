#!/usr/bin/env python
# -*- coding: utf-8 -*-

from greenness_track_toolkit.agent.core.colletor.energy.cpu.util.rapl_msr import RAPL_MSR
from greenness_track_toolkit.agent.core.colletor.energy.cpu.util.rapl_msr import RAPL_Energy
import sys


class AMD_RAPL_MSR(RAPL_MSR):
    def close(self):
        pass

    def __init__(self, cpu, debug=False):
        super().__init__()
        self._debug = debug
        self._MSR_RAPL_POWER_UNIT = 0xC0010299
        self._MSR_PKG_ENERGY_STATUS = 0xC001029B
        self._MSR_PP0_ENERGY_STATUS = 0xC001029A

        pkgs = cpu.get_pkgs()
        for pkg in pkgs:
            lead_cpuid = cpu.get_lead_cpuid(pkg)
            data = self.readmsr(self._MSR_RAPL_POWER_UNIT, lead_cpuid)
            value = 1 << ((data & self.ENERGY_UNIT_MASK) >> self.ENERGY_UNIT_OFFSET)
            # unit uJ, keeping the unit equal for sysfs powercap
            energy_unit = 1000000 / float(value)

            value = 1 << ((data & self.POWER_UNIT_MASK) >> self.POWER_UNIT_OFFSET)
            # unit uW
            power_unit = 1000000 / float(value)

            value = 1 << ((data & self.TIME_UNIT_MASK) >> self.TIME_UNIT_OFFSET)
            time_unit = 1000000 / float(value)

            self.rapl_energy[pkg] = RAPL_Energy(energy_unit, power_unit, time_unit, energy_unit, lead_cpuid)

    def __get_pkg_value(self):
        energy = {}
        for pkg, rapl in self.rapl_energy.items():
            data = self.readmsr(self._MSR_PKG_ENERGY_STATUS, rapl.lead_cpuid)
            energy[pkg] = (data & self.ENERGY_STATUS_MASK) * rapl.energy_unit
        return energy

    def __get_dram_value(self):
        sys.stderr.write("DRAM energy is not supported")
        return {}

    def start(self):
        self.last_pkg_energy = self.__get_pkg_value()
        if self._debug:
            print(self.last_pkg_energy)
        return

    def delta(self, duration):
        """
        Compute the energy used since last call.
        """
        pkg_energy = self.__get_pkg_value()
        if self._debug:
            print(pkg_energy)
        pkg_energy_delta = {}
        for pkg in self.rapl_energy.keys():
            if pkg_energy[pkg] < self.last_pkg_energy[pkg]:
                v = pkg_energy[pkg] + (self.ENERGY_STATUS_MASK - self.last_pkg_energy[pkg])
            else:
                v = pkg_energy[pkg] - self.last_pkg_energy[pkg]
            pkg_energy_delta[pkg] = v * duration

        self.last_pkg_energy = pkg_energy

        energy_sum = 0.0
        for energy in pkg_energy_delta.values():
            energy_sum += energy

        # convert unit from mj to j
        return energy_sum / 1000000.0