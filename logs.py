import logging



#   setupLogger(STRING logger_name, STRING log_file, CONST INT level)
#   sets up a logger that can easily be used like :
#   {This.ReturnedValue}.info("Info Message")
#   {This.ReturnedValue}.debug("Debug Message")
#
#
#   <Input>
#       required STRING logger_name | The string identifier for the logger.
#       required STRING log_file | The path to the log file. Can be a .txt
#       optional {CONST} INT level | The message level that is tagged with the logger.
#   <Output>
#       LOG /Anonymous/ | the logging object which can directly call log writing functions.
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

