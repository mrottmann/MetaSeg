
import os

os.system("python3 setup.py build_ext --inplace")
 
from metrics import compute_metrics, test_metrics

test_metrics()
