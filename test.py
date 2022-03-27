import copy

class C:
    def __init__(self, x):
        self.x = [e for e in range(x)]

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return str(self)

print(25.019249999999996 == 25.019249999999996)