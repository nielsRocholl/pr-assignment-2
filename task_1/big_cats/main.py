#!/bin/python3


from pipeline_modules.feature_extraction import FeatureExtractor
from pipeline_modules.clustering import Clustering


def main():
    # Classification pipeline:
    f = FeatureExtractor()
    c = Clustering(f.descriptor_list, f.num_of_samples, f.grey_dataset_flat_list, f.sift)

    # print(c.bag_of_words[10].sum())


if __name__ == '__main__':
    main()
