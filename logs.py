import logging

def setupLogger(logger_name, log_file, level = logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w+')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

    return logging.getLogger(logger_name)

''' e.g.
def main():
    setup_logger('log1', r'C:/temp/log1.log')
    setup_logger('log2', r'C:/temp/log2.log')
    log1 = logging.getLogger('log1')
    log2 = logging.getLogger('log2')

    log1.info('Info for log 1!')
    log2.info('Info for log 2!')
    log1.error('Oh, no! Something went wrong!')

'''

