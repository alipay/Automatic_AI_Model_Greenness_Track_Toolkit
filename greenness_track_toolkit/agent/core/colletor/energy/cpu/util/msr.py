#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import struct
import os


def writemsr(msr, cpuid, val):
    f = os.open('/dev/cpu/%d/msr' % cpuid, os.O_WRONLY)
    os.lseek(f, msr, os.SEEK_SET)
    os.write(f, struct.pack('Q', val))
    os.close(f)


def readmsr(msr, cpuid=0):
    f = os.open('/dev/cpu/%d/msr' % cpuid, os.O_RDONLY)
    os.lseek(f, msr, os.SEEK_SET)
    val = struct.unpack('Q', os.read(f, 8))[0]
    os.close(f)
    return val
