from utils import *

class OneHot():
    def __init__(self):
        pass
    
    def fit(self, labels, num_classes=10):
        self.num_classes = num_classes
        self.labels = labels
    
    def transform(self):
        features = np.eye(self.num_classes)[self.labels]
        return features
    
    def inverse_transform(self, features):
        labels = np.argmax(features, axis=1)
        return labels
    
    def fit_transform(self, labels, num_classes=10):
        self.fit(labels, num_classes)
        return self.transform()

class MinMax():
    def __init__(self):
        pass
    
    def normalise(self, data):
        return (data - np.min(data, axis=0)) / (np.max(data) - np.min(data, axis=0))
