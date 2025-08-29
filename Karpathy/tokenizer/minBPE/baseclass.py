# define a few helper functions and the base tokenizer class to inherit from 
# the base class is only used to load and save variables 

def get_stats(ids):
    """returns a dict containing pair:count mapping"""
    
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair , 0) + 1 # fetch existing value or 0 and add if pair is encountered

    return counts


def merge(ids, pair, idx):
    """Implement the BPE algorithm by merging the top_pair in ids by idx. Append top_pair:idx to vocab dictionary"""

    new_ids = []
    i = 0
    while i < len(ids): # not <=, since +1 is already done
        if i < len(ids)-1 and (ids[i], ids[i+1]) == (pair[0], pair[1]) :
            new_ids.append(idx)
            i+=2    
        else:
            new_ids.append(ids[i])
            i +=1
    
    return new_ids



ids = [1,2,-1,0, 4, 1,2, 1,2]
print(get_stats(ids=ids))
print(merge(ids, (1,2), 256))