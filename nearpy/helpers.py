# Ensures logger handlers only show messages at a given level 
class LoggingFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level