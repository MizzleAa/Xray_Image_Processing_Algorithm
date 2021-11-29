import logging
import sys
import os 

abs_path = os.path.abspath('.')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

format= logging.Formatter('%(asctime)s [ %(levelname)s ] - %(filename)s : %(lineno)s >> %(message)s')

stream_handler = logging.StreamHandler()

file_handler = logging.FileHandler(f"{abs_path}/log/state.log", encoding="utf-8")

stream_handler.setFormatter(format)
file_handler.setFormatter(format)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

def logger_exception(ex, memo=""):
    exc_space = " "*24
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logger.warn(f"\n{exc_space}exc_type = {exc_type}\n{exc_space}file_name = {file_name}\n{exc_space}tb_lineno = {exc_tb.tb_lineno}\n{exc_space}exc_obj = {exc_obj}\nmemo={memo}")
