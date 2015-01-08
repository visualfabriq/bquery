from vbench.api import Benchmark
from datetime import datetime

common_setup = """
import bquery
"""

setup = common_setup + """
import time
"""

stmt2 = "time.sleep(1)"
bm_groupby2 = Benchmark(stmt2, setup, name="GroupBy test 1",
                        start_date=datetime(2011, 7, 1))
