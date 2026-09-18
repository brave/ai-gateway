from __future__ import annotations

import ctypes
import os
import platform
import struct

SYS_landlock_create_ruleset = 444
SYS_landlock_add_rule = 445
SYS_landlock_restrict_self = 446

LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1

PR_SET_NO_NEW_PRIVS = 38

AL_FS_EXECUTE = 1 << 0
AL_FS_WRITE_FILE = 1 << 1
AL_FS_READ_FILE = 1 << 2
AL_FS_READ_DIR = 1 << 3
AL_FS_REMOVE_DIR = 1 << 4
AL_FS_REMOVE_FILE = 1 << 5
AL_FS_MAKE_CHAR = 1 << 6
AL_FS_MAKE_DIR = 1 << 7
AL_FS_MAKE_REG = 1 << 8
AL_FS_MAKE_SOCK = 1 << 9
AL_FS_MAKE_FIFO = 1 << 10
AL_FS_MAKE_BLOCK = 1 << 11
AL_FS_MAKE_SYM = 1 << 12
AL_FS_REFER = 1 << 13
AL_FS_TRUNCATE = 1 << 14

AL_NET_BIND_TCP = 1 << 0
AL_NET_CONNECT_TCP = 1 << 1

ABI_HANDLED_ACCESS = {
    1: 0x3FFF,
    2: 0x3FFF | AL_FS_REFER,
    3: 0x3FFF | AL_FS_REFER | AL_FS_TRUNCATE,
}

AL_NET_HANDLED = AL_NET_BIND_TCP | AL_NET_CONNECT_TCP

_RULESET_ATTR_FS = struct.Struct("=Q")
_RULESET_ATTR_FS_NET = struct.Struct("=QQ")

READ_ACCESS = AL_FS_EXECUTE | AL_FS_READ_FILE | AL_FS_READ_DIR

_RULESET_ATTR = struct.Struct("=Q")
_PATH_BENEATH = struct.Struct("=Qi4x")


class LandlockUnavailable(RuntimeError):
    pass


def probe_abi() -> int:
    if platform.system() != "Linux":
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    ret = libc.syscall(
        ctypes.c_long(SYS_landlock_create_ruleset),
        ctypes.c_void_p(0),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if ret < 0:
        return 0
    return int(ret)


def apply_landlock(
    read_paths: list[str], rw_paths: list[str], abi: int | None = None
) -> int:
    if platform.system() != "Linux":
        raise LandlockUnavailable("Landlock is Linux-only")

    if abi is None:
        abi = probe_abi()
    if abi < 1:
        raise LandlockUnavailable("Kernel does not support Landlock (>= 5.13 needed)")

    handled_fs = ABI_HANDLED_ACCESS[min(abi, 3)]
    handled_net = AL_NET_HANDLED if abi >= 4 else 0
    if abi >= 4:
        ruleset_attr = _RULESET_ATTR_FS_NET.pack(handled_fs, handled_net)
    else:
        ruleset_attr = _RULESET_ATTR_FS.pack(handled_fs)
    write_access = (
        AL_FS_READ_FILE
        | AL_FS_READ_DIR
        | AL_FS_WRITE_FILE
        | AL_FS_MAKE_REG
        | AL_FS_MAKE_DIR
        | AL_FS_REMOVE_FILE
        | AL_FS_REMOVE_DIR
        | AL_FS_MAKE_SYM
    ) & handled_fs

    libc = ctypes.CDLL(None, use_errno=True)

    rc = libc.prctl(
        ctypes.c_int(PR_SET_NO_NEW_PRIVS),
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if rc != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")

    ctypes.set_errno(0)
    ruleset_fd = libc.syscall(
        ctypes.c_long(SYS_landlock_create_ruleset),
        ctypes.c_char_p(ruleset_attr),
        ctypes.c_size_t(len(ruleset_attr)),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    try:

        def _add_rule(path: str, access: int) -> None:
            o_path = getattr(os, "O_PATH", 0)
            fd = os.open(path, o_path | os.O_CLOEXEC)
            try:
                packed = _PATH_BENEATH.pack(access, fd)
                ctypes.set_errno(0)
                rc = libc.syscall(
                    ctypes.c_long(SYS_landlock_add_rule),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.c_char_p(packed),
                    ctypes.c_uint(0),
                )
                if rc != 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule({path})")
            finally:
                os.close(fd)

        for p in read_paths:
            _add_rule(p, READ_ACCESS)
        for p in rw_paths:
            _add_rule(p, write_access)

        ctypes.set_errno(0)
        rc = libc.syscall(
            ctypes.c_long(SYS_landlock_restrict_self),
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
        if rc != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)

    return abi
