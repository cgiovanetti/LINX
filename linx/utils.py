"""Miscellaneous LINX runtime utilities."""

import ctypes
import gc
import sys


def release_unused_memory():
    """Return freed heap pages to the OS.

    JAX/XLA compile leaves several hundred MiB of freed-but-untrimmed memory
    on systems with glibc. Calling this once -- typically right after
    the first BackgroundModel / AbundanceModel call -- can free up to ~half
    of the memory needed per solve.

    Safe to call any time, cannot drop live memory. 
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
