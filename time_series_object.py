class TimeSeriesObject:
    def __init__(self, classification, gold, xp, cs):
        self.classification = classification
        self.gold = gold
        self.xp = xp
        self.cs = cs

    def getFirstFifteenMinutesGold(self):
        return self.gold[:14]

    def getFirstFifteenMinutesXp(self):
        return self.xp[:14]

    def getFirstFifteenMinutesCs(self):
        return self.xp[:14]
