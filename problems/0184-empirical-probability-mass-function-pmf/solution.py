from collections import defaultdict

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    # maybe dict to count how often each outcome appears
    d = defaultdict(int)
    for i in range(len(samples)):
        d[samples[i]] += 1
    
    answer = []
    for (val, count) in d.items(): # * d.items
        answer.append((val, count / len(samples)))
    return answer