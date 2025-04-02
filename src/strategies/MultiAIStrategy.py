# MultiAIStrategy.py
from freqtrade.strategy import IStrategy

class MultiAIStrategy(IStrategy):
    def populate_indicators(self, dataframe, metadata):
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[dataframe['rsi'] > 70, 'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[dataframe['rsi'] < 30, 'exit_long'] = 1
        return dataframe
