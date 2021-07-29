import random
import numpy as np

class Dataset:
    def __init__(self, data):
        self.raw = data
        self.perturbed = data
        self.removalPerAmount = len(self.raw[0]) / 100

        # Initialize perturbation values
        self.jitter = 0
        self.translateDims = []
        self.translateDimAmount = 0
        self.translate_value = 0
        self.scaleDims = []
        self.scaleDimAmount = 0
        self.local_scale_value = 1
        self.global_scale_value = 1
        self.permuteDims = []
        self.nonActiveDims = []

        print("Dataset object created")

    def combinePerturbations(self):
        intermediate = self.raw.copy()

        # Handle translation
        for i in self.translateDims:
            intermediate[:, i] += self.translate_value

        # Handle jitter
        intermediate += self.jitter

        # Handle local scaling
        for i in self.scaleDims:
            intermediate[:, i] *= self.local_scale_value

        # Compute the difference with the original
        delta = intermediate - self.raw
        # Scale the difference and add to original
        intermediate = self.raw + delta * self.global_scale_value

        # Normalize array to range (0, 1)
        arr_max = np.amax(intermediate)
        arr_min = np.amin(intermediate)
        self.perturbed = (intermediate - arr_min) / (arr_max - arr_min)

        # For each dimension, randomly choose another dimension to switch values with
        for i in self.permuteDims:
            # Make sure not the same index is picked
            j = i
            while i == j:
                j = random.randint(0, len(self.raw[0]) - 1)

            tmp = self.perturbed[:, i]
            self.perturbed[:, i] = self.perturbed[:, j]
            self.perturbed[:, j] = tmp

        # Handle non-active dimensions
        for i in self.nonActiveDims:
            self.perturbed[:, i] = 0

    def adjustRandomDimensions(self, number, dimension_list):
        amountToRemove = number * self.removalPerAmount
        currentRemovalCount = len(dimension_list)

        if currentRemovalCount == amountToRemove:
            return dimension_list
        elif currentRemovalCount < amountToRemove:
            transDims = [x for x in [*range(0, len(self.raw[0]))] if x not in dimension_list]
            amountToAdd = amountToRemove - currentRemovalCount
            newRandoms = random.sample(transDims, int(amountToAdd))
            dimension_list.extend(newRandoms)
        else:
            numberToRemove = int(currentRemovalCount - amountToRemove)
            dimension_list = dimension_list[:len(dimension_list) - numberToRemove]
        return dimension_list

    # Add a constant value to randomly chosen dimensions
    def translation(self, amount, dims):
        # Choose random dimensions to translate
        self.translateDims = self.adjustRandomDimensions(dims, self.translateDims)

        self.translate_value = 0.01 * amount
        self.combinePerturbations()

    # Scale randomly chosen dimensions
    def scale(self, amount, dims):
        # Choose random dimensions on which to scale
        self.scaleDims = self.adjustRandomDimensions(dims, self.scaleDims)

        self.local_scale_value = 0.01 * amount
        self.combinePerturbations()

    # Randomly choose 2 dimensions and switch their values
    def permute(self, amount):
        self.permuteDims = self.adjustRandomDimensions(amount, self.permuteDims)

        self.combinePerturbations()

    # Randomly set dimensions to non-active
    def removeRandomDimensions(self, amount):
        self.nonActiveDims = self.adjustRandomDimensions(amount, self.nonActiveDims)

        self.combinePerturbations()

    # Add random jitter, intensity means how much jitter
    def jitterNoise(self, intensity):
        if intensity == 0:
            self.jitter = 0
        else:
            # Clip intensity to a value between 0 and 1
            intensity = max(-1.0, min(1.0, float(intensity)))
            self.jitter = np.random.uniform(-intensity, intensity, size=(len(self.raw), len(self.raw[0])))

        self.combinePerturbations()

    def scaleAllPerturbations(self, amount):
        self.global_scale_value = 0.01 * amount
        self.combinePerturbations()

    def perturbAll(self, amount):
        pass

    # For a chosen perturbation
    # Compute the intermediate datasets from 0 to given value and add them to a list
    def interDataOfPerturb(self, perturbation, maxValue):
        self.interDataset = []
        print(f"Perturbation: {perturbation}")
        for i in range(0, maxValue):
            # match token coming in python 3.10
            if perturbation == 0:
                self.perturbAll(i)
            elif perturbation == 1:
                self.translation(i, self.translateDimAmount)
            elif perturbation == 2:
                self.translation(self.translate_value, i)
            elif perturbation == 3:
                self.scale(i, self.scaleDimAmount)
            elif perturbation == 4:
                self.scale(self.local_scale_value, i)
            elif perturbation == 5:
                self.jitterNoise(0.1)
            elif perturbation == 6:
                self.scaleAllPerturbations(maxValue - i)
            elif perturbation == 7:
                self.permute(maxValue)
            elif perturbation == 8:
                self.removeRandomDimensions(i)
            else:
                print("No perturbation found with index " + str(i))
            self.interDataset.append(self.perturbed)
