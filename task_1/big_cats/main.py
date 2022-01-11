#!/bin/python3


from pipeline_modules.feature_extraction import FeatureExtractor
from pipeline_modules.clustering import Clusterer 

f = FeatureExtractor()
Clusterer(f.processed_dataset)