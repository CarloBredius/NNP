class Dataset:
    def __init__(self, data):
        self.raw = data
        self.perturbed = data
        print("Dataset object created")

    def addNoise(self, amount):
        #print("Adding " + str(amount) + " noise to: " + str(self.raw))
        self.perturbed = self.raw + 0.005 * amount
