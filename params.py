class Params():

    def __init__(self, d=None):
        if d:
            self.from_dict(d)

    def from_dict(self, d):
        for attr, value in d.items():
            setattr(self, attr, value)

    # def values(self):
    #     
