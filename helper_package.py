import numpy as np

class OneHot():
    def __init__(self):
        pass
    
    def fit(self, labels, num_classes):
        self.num_classes = num_classes
        self.labels = labels
    
    def transform(self):
        features = np.eye(self.num_classes)[self.labels]
        return features
    
    def inverse_transform(self, features):
        labels = np.argmax(features, axis=1)
        return labels
    
    def fit_transform(self, labels, num_classes):
        self.fit(labels, num_classes)
        return self.transform()

def min_max_scale(x):
    