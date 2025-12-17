import logging
import os
import sys
import warnings


# Logger setup
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sampling'))
sys.path.insert(0, module_path)
from misc import *


logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s  (%(filename)s:%(lineno)d)")
fileHandler = logging.FileHandler("log.txt", mode='w')
fileHandler.setFormatter(logFormatter)
consoleHandler = logging.StreamHandler()
consoleFormatter = CustomFormatter()
consoleHandler.setFormatter(consoleFormatter)
log = logging.getLogger()
for hdlr in log.handlers[:]:
    log.removeHandler(hdlr)
log.addHandler(fileHandler)
log.addHandler(consoleHandler)
log.setLevel(logging.INFO)

# Warning filters
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="tensorflow.python.data.ops.structured_function"
)
