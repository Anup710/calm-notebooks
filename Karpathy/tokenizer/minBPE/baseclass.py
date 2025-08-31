# define a few helper functions and the base tokenizer class to inherit from 
# the base class is only used to load and save variables 
# the helper functions are picked up directly from karpathy's minbpe as i wasnt aware of 'control chars' 
# beforehand

import unicodedata 

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


# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


class BaseTokenizer():

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        # start with a base vocab of 256 bytes 
        self.vocab = self._build_vocab()
        self.dummy = "Hi from parent class"

    def train(self, text, num_merges, verbose):
        # trains the tokenizer on text by implement BPE for num_merges merges 
        # returns self.merges: dict, self.new_vocab: dict
        # fill this as per the tokenizer type
        raise NotImplementedError
    
    def encode(self, input_text) -> list:
        # returns encoded text
        raise NotImplementedError
    
    def decode(self, ids) -> str:
        # accepts list of integers and returns str
        raise NotImplementedError
    
    def _build_vocab(self):
        # return {"hi":"there"}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for pair, idx in self.merges.items():
            vocab[idx] = (pair[0], pair[1])
        
        for spec, idx in self.special_tokens.items():
            vocab[idx] = spec.encode("utf-8")
        return vocab
    
    # save and load copied as it is from minbpe
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

    


# mod_dict = {(1,2): 10, (113,257): 90, (-1,4):30}
# for pair, count in mod_dict.items():
#     print(pair[0]+pair[1])

if __name__ == "__main__":
    
    # check if merge works
    ids = [1,2,-1,0, 4, 1,2, 1,2]
    print(get_stats(ids=ids))
    print(merge(ids, (1,2), 256))

    # check if vocab is correctly initialized
    tok = BaseTokenizer()
    print(tok.vocab)