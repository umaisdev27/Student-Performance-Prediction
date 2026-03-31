# any execution that happens, we should we able to log those info in some files
# through tracking  

import logging
import os
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

log_file_path = os.path.join(logs_path, log_file)

logging.basicConfig(
    filename=log_file_path,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__=="__main__":
    logging.info("we doing logging")