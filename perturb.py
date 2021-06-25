import random
import numpy as np

class Dataset:
    def __init__(self, data):
        self.raw = data
        self.jitter = 0
        self.noise = 0
        self.perturbed = data
        self.nonActiveDims = []
        self.removalPerAmount = len(self.raw[0]) / 100
        print("Dataset object created")

    def combinePerturbations(self):
        # Add jitter and constant noise, clipped to be between 0 and 1
        self.perturbed = np.clip(self.raw + self.jitter + self.noise, 0, 1)

        # Handle non-active dimensions
        for i in self.nonActiveDims:
            self.perturbed[:, i] = 0

    # Add random jitter, intensity means how much jitter
    def jitterNoise(self, intensity):
        if intensity == 0:
            self.jitter = 0
        else:
            # Clip intensity to a value between 0 and 1
            intensity = max(-1.0, min(1.0, float(intensity)))
            self.jitter = np.random.uniform(-intensity, intensity, size=(len(self.raw), len(self.raw[0])))

        self.combinePerturbations()

    def addConstantNoise(self, amount):
        self.noise = 0.01 * amount
        self.combinePerturbations()

    # Randomly set dimensions to 0. Potentailly chooses a dimensions that is already 0
    def removeRandomDimensions(self, amount):        
        amountToRemove = amount * self.removalPerAmount
        currentRemovalCount = len(self.nonActiveDims)
        if currentRemovalCount < amountToRemove:
            activeDims = [x for x in [*range(0, len(self.raw[0]))] if x not in self.nonActiveDims]
            amountToAdd = amountToRemove - currentRemovalCount
            newRandoms = random.sample(activeDims, int(amountToAdd))
            self.nonActiveDims.extend(newRandoms)
        else:
            numberToRemove = int(currentRemovalCount - amountToRemove)
            self.nonActiveDims = self.nonActiveDims[:len(self.nonActiveDims) - numberToRemove]

        self.combinePerturbations()

    def perturbAll(self, amount):
        pass

    # For a chosen perturbation
    # Compute the intermediate datasets from 0 to given value and add them to a list
    def interDataOfPerturb(self, perturbation, maxValue):
        self.interDataset = []

        for i in range(0, maxValue):
            # match token coming in python 3.10
            if perturbation == 0:
                self.perturbAll(i)
            elif perturbation == 1:
                self.addConstantNoise(i)
            elif perturbation == 2:
                self.removeRandomDimensions(i)
            elif perturbation == 3:
                self.jitterNoise(0.1)
            else:
                print("No perturbation found with index " + str(i))
            self.interDataset.append(self.perturbed)
