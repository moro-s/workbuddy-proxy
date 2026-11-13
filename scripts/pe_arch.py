#!/usr/bin/env python3
"""Assert Windows executables were built for the expected architecture.

The machine field is read from each PE header so the check does not depend on a
toolchain being on PATH: a package labelled x64 must contain x64 binaries.
"""
import argparse
import struct
import sys

MACHINES = {0x8664: 'x64', 0x014c: 'x86', 0xaa64: 'arm64'}


def read_machine(path):
    with open(path, 'rb') as f:
        dos = f.read(0x40)
        if len(dos) < 0x40 or dos[:2] != b'MZ':
            raise SystemExit(f'{path}: not a PE file')
        pe = struct.unpack_from('<I', dos, 0x3C)[0]
        f.seek(pe)
        if f.read(4) != b'PE\0\0':
            raise SystemExit(f'{path}: missing PE signature')
        return struct.unpack('<H', f.read(2))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect', default='x64', choices=sorted(MACHINES.values()))
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    failed = False
    for path in args.files:
        machine = read_machine(path)
        name = MACHINES.get(machine, f'0x{machine:04x}')
        status = 'ok' if name == args.expect else f'expected {args.expect}'
        print(f'{path}: {name} ({status})')
        failed |= name != args.expect
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
