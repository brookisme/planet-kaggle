
###################################################################
#
# MEASURE_BASE: USE WHEN CREATING HISTORY CALLBACKS 
#
####################################################################
class HistoryMeasures(object):
    """ init
        Args:
            properties <list>: properties to track from 
            the logs dict in history object
    """
    def __init__(self,properties):
        self.properties=properties
        for prop in self.properties:
            setattr(self,prop,[])


    def update(self,logs):
        logs=logs.copy()
        """ update properties:
            logs: the logs dict in history callbacks
        """
        for prop in self.properties:
            value=logs.pop(prop,None)
            if value:
                getattr(self, prop).append(value)


    def dict(self):
        """ get current values as dictionary
        """
        measures={}
        for prop in self.properties:
            measures[prop]=getattr(self,prop)
        return measures