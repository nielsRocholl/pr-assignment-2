from cv2 import kmeans


class Clusterer:
    def __init__(self, data):
        self.data = data
        self.clusters = {}
        print(self.__cluster_data())
        
    
    def __cluster_data(self):
        return kmeans(self.data, 8)
