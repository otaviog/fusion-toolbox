"""Wraps doctest execution"""

import doctest

if __name__ == "__main__":
    import fusionkit._open3d_functions
    doctest.testmod(fusionkit._open3d_functions)

