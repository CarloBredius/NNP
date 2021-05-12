import random

class Dataset:
    def __init__(self, data):
        self.raw = data
        self.noise = data
        self.perturbed = data

        self.zeroWeightIndices = []
        self.removalPerAmount = len(self.raw[0]) / 100
        print("Dataset object created")

    def combinePerturbations(self):
        self.perturbed = self.raw + self.noise
        for i in self.zeroWeightIndices:
            self.perturbed[:, i] = 0

    def addConstantNoise(self, amount):
        self.noise = 0.005 * amount
        self.combinePerturbations()

    # Randomly set dimension weights to 0. The chosen dimensions are random,
    # so each time the slider is higher, it will choose different dimensions
    # Todo: discuss whether we want a less ambiguous approach
    def removeRandomDimensions(self, amount):
        amountToRemove = amount * self.removalPerAmount
        currentRemovalCount = len(self.zeroWeightIndices)
        if currentRemovalCount < amountToRemove:
            amountToAdd = amountToRemove - currentRemovalCount
            newRandoms = random.sample(range(0, len(self.raw[0])), int(amountToAdd))
            self.zeroWeightIndices.extend(newRandoms)
        else:
            numberToRemove = int(currentRemovalCount - amountToRemove)
            self.zeroWeightIndices = self.zeroWeightIndices[:len(self.zeroWeightIndices) - numberToRemove]

        self.combinePerturbations()


