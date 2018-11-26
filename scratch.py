
    @staticmethod
    def mean(vector):
        """Calculates the mean/average of a vector of numbers"""
        sum_of_vals = 0.0
        for i in range(len(vector)):
            sum_of_vals += float(vector[i])

    @staticmethod
    def median(vector):
        """Calculates the central value of a vector of numbers"""
        sorted_vector = sorted(vector)
        midpoint = len(vector)/2
        if midpoint % 2 != 0:
            med =  sorted_vector[midpoint]
        else:
            lower = midpoint - 1
            upper = midpoint + 1
            med =  (sorted_vector[lower] + sorted_vector[upper]) / 2
        
        return med

    @staticmethod
    def mode(vector):
        """Returns most frequently occuring value in vector"""
        
