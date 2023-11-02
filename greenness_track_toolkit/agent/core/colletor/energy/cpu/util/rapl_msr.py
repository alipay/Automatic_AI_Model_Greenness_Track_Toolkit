#!/usr/bin/env python
# -*- coding: utf-8 -*-

import greenness_track_toolkit.agent.core.colletor.energy.cpu.util.msr as msr
from greenness_track_toolkit.agent.core.colletor.collector_base import Collector


class RAPL_Energy:
    def __init__(
        self,
        energy_unit,
        power_unit,
        time_unit,
        dram_unit,
        lead_cpuid
    ):
        self.energy_unit = energy_unit
        self.power_unit = power_unit
        self.time_unit = time_unit
        self.dram_unit = dram_unit
        self.lead_cpuid = lead_cpuid


class RAPL_MSR(Collector):
    def __init__(self):
        super().__init__()
        self.ENERGY_UNIT_MASK = 0x1F00
        self.ENERGY_UNIT_OFFSET = 0x08
        self.POWER_UNIT_MASK = 0x0F
        self.POWER_UNIT_OFFSET = 0x0
        self.TIME_UNIT_MASK = 0xF000
        self.TIME_UNIT_OFFSET = 0x10
        self.ENERGY_STATUS_MASK = 0xFFFFFFFF
        self.rapl_energy = {}

    def readmsr(self, msr_addr, cpuid):
        return msr.readmsr(msr_addr, cpuid)

    def writemsr(slef, msr_addr, cpuid, value, print_only=False):
        if print_only:
            return
        msr.writemsr(msr_addr, cpuid, value)