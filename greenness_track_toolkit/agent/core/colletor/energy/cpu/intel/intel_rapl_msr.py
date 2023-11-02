#!/usr/bin/env python
# -*- coding: utf-8 -*-

from greenness_track_toolkit.agent.core.colletor.energy.cpu.util.rapl_msr import RAPL_MSR
from greenness_track_toolkit.agent.core.colletor.energy.cpu.util.rapl_msr import RAPL_Energy

CORE_NAME = "core_name"

DRAM_ENERGY = "dram_energy"
DRAM_UNIT = "dram_unit"

cpu_mapping = {
    0x2A: {CORE_NAME: "sandybridge", DRAM_ENERGY: False, DRAM_UNIT: False},
    0x2D: {CORE_NAME: "sandybridge-x", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x3A: {CORE_NAME: "ivybridge", DRAM_ENERGY: False, DRAM_UNIT: False},
    0x3E: {CORE_NAME: "ivybridge-x", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x3C: {CORE_NAME: "haswell", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x3F: {CORE_NAME: "haswell-x", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x45: {CORE_NAME: "haswell-l", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x46: {CORE_NAME: "haswell-g", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x3D: {CORE_NAME: "broadwell", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x47: {CORE_NAME: "broadwell-g", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x4F: {CORE_NAME: "broadwell-x", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x56: {CORE_NAME: "broadwell-d", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x57: {CORE_NAME: "knights-l", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x85: {CORE_NAME: "knights-m", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x4E: {CORE_NAME: "skylake-l", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x5E: {CORE_NAME: "skylake", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x55: {CORE_NAME: "skylake-x", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x8E: {CORE_NAME: "kabylake-l", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x9E: {CORE_NAME: "kabylake", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x66: {CORE_NAME: "cannonlake-l", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x5C: {CORE_NAME: "goldmont", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x5F: {CORE_NAME: "goldmont-d", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x7A: {CORE_NAME: "goldmont-p", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x6A: {CORE_NAME: "icelake-x", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x6C: {CORE_NAME: "icelake-d", DRAM_ENERGY: True, DRAM_UNIT: True},
    0x7D: {CORE_NAME: "icelake", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x7E: {CORE_NAME: "icelake-l", DRAM_ENERGY: True, DRAM_UNIT: False},
    0xA5: {CORE_NAME: "cometlake", DRAM_ENERGY: True, DRAM_UNIT: False},
    0xA6: {CORE_NAME: "cometlake-l", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x97: {CORE_NAME: "alderlake", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x9A: {CORE_NAME: "alderlake-l", DRAM_ENERGY: True, DRAM_UNIT: False},
    0x8F: {CORE_NAME: "sapphirerapids-x", DRAM_ENERGY: True, DRAM_UNIT: False},

}


class INTEL_RAPL_MSR(RAPL_MSR):
    def close(self):
        pass

    def __init__(self, cpu, debug=False):

        super().__init__()
        self._debug = debug
        self._MSR_RAPL_POWER_UNIT = 0x606
        self._MSR_PKG_ENERGY_STATUS = 0x611

        self._MSR_DRAM_ENERGY_STATUS = 0x619
        self._cpu_model = None
        self.rapl_energy = {}

        if cpu.vendor == "GenuineIntel" and cpu.family == 6:
            if cpu.model in cpu_mapping:
                self._cpu_model = cpu_mapping[cpu.model]

        if self._cpu_model is None:
            raise Exception("CPU family %d not supported" % cpu.model)

        pkgs = cpu.get_pkgs()
        for pkg in pkgs:
            lead_cpuid = cpu.get_lead_cpuid(pkg)
            data = self.readmsr(self._MSR_RAPL_POWER_UNIT, lead_cpuid)
            value = 1 << ((data & self.ENERGY_UNIT_MASK) >> self.ENERGY_UNIT_OFFSET)
            energy_unit = 1000000 / float(value)

            value = 1 << ((data & self.POWER_UNIT_MASK) >> self.POWER_UNIT_OFFSET)
            power_unit = 1000000 / float(value)

            value = 1 << ((data & self.TIME_UNIT_MASK) >> self.TIME_UNIT_OFFSET)
            time_unit = 1000000 / float(value)

            dram_unit = energy_unit
            if self._cpu_model[DRAM_UNIT]:
                dram_unit = 1000000 / float(1 << 16)

            self.rapl_energy[pkg] = RAPL_Energy(energy_unit, power_unit, time_unit, dram_unit, lead_cpuid)

    def __get_pkg_value(self):
        energy = {}
        for pkg, rapl in self.rapl_energy.items():
            data = self.readmsr(self._MSR_PKG_ENERGY_STATUS, rapl.lead_cpuid)
            energy[pkg] = (data & self.ENERGY_STATUS_MASK)
        return energy

    def __get_dram_value(self):
        energy = {}
        if self._cpu_model[DRAM_ENERGY]:
            for pkg, rapl in self.rapl_energy.items():
                data = self.readmsr(self._MSR_DRAM_ENERGY_STATUS, rapl.lead_cpuid)
                energy[pkg] = (data & self.ENERGY_STATUS_MASK)
        return energy

    def start(self):
        self.last_pkg_energy = self.__get_pkg_value()
        self.last_dram_energy = self.__get_dram_value()
        if self._debug:
            print(self.last_pkg_energy)
            print(self.last_dram_energy)
        return

    def delta(self, duration):
        """
        Compute the energy used since last call.
        """
        pkg_energy = self.__get_pkg_value()
        dram_energy = self.__get_dram_value()
        if self._debug:
            print(pkg_energy)
            print(dram_energy)
        pkg_energy_delta = {}
        dram_energy_delta = {}
        for pkg, rapl in self.rapl_energy.items():
            v = 0.0
            if pkg_energy[pkg] < self.last_pkg_energy[pkg]:
                v = pkg_energy[pkg] + (self.ENERGY_STATUS_MASK - self.last_pkg_energy[pkg])
            else:
                v = pkg_energy[pkg] - self.last_pkg_energy[pkg]
            pkg_energy_delta[pkg] = v * rapl.energy_unit * duration

            if self._cpu_model[DRAM_ENERGY]:
                if dram_energy[pkg] < self.last_dram_energy[pkg]:
                    v = dram_energy[pkg] + (self.ENERGY_STATUS_MASK - self.last_dram_energy[pkg])
                else:
                    v = dram_energy[pkg] - self.last_dram_energy[pkg]
                dram_energy_delta[pkg] = v * rapl.dram_unit * duration

        self.last_pkg_energy = pkg_energy
        self.last_dram_energy = dram_energy

        energy_sum = 0.0
        for energy in pkg_energy_delta.values():
            energy_sum += energy
        for energy in dram_energy_delta.values():
            energy_sum += energy
        return energy_sum / 1000000.0
