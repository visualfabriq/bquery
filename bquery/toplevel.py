import os

from bcolz.ctable import ROOTDIRS

import bquery


def open(rootdir, mode='a'):
    # ----------------------------------------------------------------------
    # https://github.com/Blosc/bcolz/blob/master/bcolz/toplevel.py#L104-L132
    # ----------------------------------------------------------------------
    """
    open(rootdir, mode='a')

    Open a disk-based carray/ctable.
    This function could be used to open bcolz objects as bquery objects to
    perform queries on them.

    Parameters
    ----------
    rootdir : pathname (string)
        The directory hosting the carray/ctable object.
    mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns
    -------
    out : a carray/ctable object or IOError (if not objects are found)

    """
    # First try with a carray
    rootsfile = os.path.join(rootdir, ROOTDIRS)
    if os.path.exists(rootsfile):
        return bquery.ctable(rootdir=rootdir, mode=mode)
    else:
        return bquery.carray(rootdir=rootdir, mode=mode)
