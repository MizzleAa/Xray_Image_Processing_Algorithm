from library.utils.header import *
from library.utils.log import *

def dec_func_start_end(func):
    def wrapper(*args, **kwargs):
        logger.info(f"{func.__name__} - start")
        start_time = datetime.datetime.now().timestamp()
        func(*args, **kwargs)
        end_time = datetime.datetime.now().timestamp()
        fts = datetime.datetime.fromtimestamp(end_time-start_time)
        fts = fts.strftime("%H:%M:%S.%f")
        logger.info(f"{func.__name__} - end , [fts : {fts}]")
    return wrapper
