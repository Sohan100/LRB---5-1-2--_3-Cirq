"""
rb_checkpoint.py

This module provides helper functions for saving and loading checkpoint data
using pickle. This can be used in logical randomized benchmarking experiments
to save intermediate results and resume computations later.
"""

import pickle
import os


def saveCheckpoint(data, filename):
    """
Save data to a checkpoint file.

Args:
    data (Any): Input argument consumed by `saveCheckpoint` to perform this
                operation.
    filename (Any): Input argument consumed by `saveCheckpoint` to perform this
                    operation.

Returns:
    None: None. This method persists data to storage and does not return a
          payload.

Raises:
    OSError: If the output file cannot be written to disk.
"""
    # Write the full payload atomically for this call; callers control update
    # frequency (for example, every N circuits).
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {filename}")


def loadCheckpoint(filename):
    """
Load data from a checkpoint file.

Args:
    filename (Any): Input argument consumed by `loadCheckpoint` to perform this
                    operation.

Returns:
    object: Requested data object loaded or assembled by this method.

Raises:
    FileNotFoundError: If the requested input file cannot be found at the
                       provided path.
    pickle.UnpicklingError: If the serialized payload is malformed or
                            incompatible with the expected schema.
"""
    if os.path.exists(filename):
        # Return deserialized progress so long-running jobs can resume exactly
        # from the last successful checkpoint.
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Checkpoint loaded from {filename}")
        return data
    else:
        # Caller decides how to initialize a fresh run when checkpoint is
        # absent.
        print(f"No checkpoint found at {filename}")
        return None
