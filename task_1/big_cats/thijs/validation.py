import random

class K_folds:
    def __init__(self, data, k=10):
        random.seed(0)
        self.cur_fold = 0
        random.shuffle(data)
        self.fold = [[dp for dp in data[idx::k]] for idx in range(k)]
        random.shuffle(self.fold)

    def __iter__(self):
        """
        Generates k training and test sets.
        """
        for idx in range(len(self.fold)):
            test_set = self.fold[idx]
            training_set = [item for idx2 in range(len(self.fold)) if idx2 != idx for item in self.fold[idx2]]
            yield training_set, test_set

