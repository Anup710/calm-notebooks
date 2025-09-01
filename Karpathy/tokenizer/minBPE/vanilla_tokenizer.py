# a regular tokenizer class based on the BPE algorithm by Seinrich et al. 

from baseclass import BaseTokenizer, render_token, get_stats, merge


class RegularTokenizer(BaseTokenizer):

    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose: bool):
        """"
        Trains the Tokenizer on an input text and performs merges for {num_merges} 
        iterations, adds merges to the merges dict. 
        returns: merges: dict, vocab: dict 
        """
        assert vocab_size >=256

        num_merges = vocab_size - 256
        enc_text = text.encode("utf-8")
        enc_ids = list(map(int, enc_text))

        # create internal base vocab and internal merges within train
        in_merges = {} 
        in_vocab = {i : bytes([i]) for i in range(256)}

        for m in range (num_merges):
            # dict of pair: counts
            stats = get_stats(enc_ids)
            # extract pair with most counts
            max_pair = max(stats, key=stats.get)
            
            # idx
            idx = 256 + m
            #merge
            enc_ids = merge(enc_ids, max_pair, idx)
            
            # append max pair to in_merges and in_vocab
            in_merges[max_pair] = 256 + m
            in_vocab[idx] = in_vocab[max_pair[0]] + in_vocab[max_pair[1]]  # Concatenate the bytes
            if verbose:
                print(f"merge {m+1}/{num_merges}: {max_pair} -> {idx} ({in_vocab[idx]}) had {stats[max_pair]} occurrences")

        # transfer learnt variables to the class. 
        self.merges = in_merges
        self.vocab = in_vocab

    
    # inference 

    def encode(self, text):
        """"returns tokens after applying learnt merges on text during inference"""
        
        tokens = list(text.encode("utf-8"))
        print(f"Length before BPE merge = {len(tokens)}")

        while len(tokens) >= 2: 
            stats = get_stats(tokens)
            # min over keys of dict stats, key = index of that key pair; "inf" applied to ensure merges
            # progress in correct order of occurance
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) 
            if pair not in self.merges:
                break # nothing else can be merged
            # else merge whatever is available
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        
        print(f"Length after BPE merge = {len(tokens)}")
        
        # return tokens   


    def decode(self, ids):
        """ returns the original string by using self.vocab on a list of integers"""
        # convert int to bytes using vocab, since render_tokens works on bytes
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        s = render_token(text_bytes)
        return s


        

if __name__ == "__main__":
    # test code
    # print(__name__)
    reg = RegularTokenizer()
    print(reg.dummy)

    with open("tests/novakdjokovic.txt", "r", encoding="utf-8") as f1:
        training_text = f1.read()
        
    RegTok = RegularTokenizer()
    RegTok.train(training_text, 300, True)

    # lets encode the taylorswift.txt file
    with open("tests/taylorswift.txt", "r", encoding="utf-8") as f2:
        inference_text = f2.read()
    
    # print(f"Encoded text:\n{RegTok.encode(inference_text)}")
    RegTok.encode(inference_text)
    print(RegTok.vocab)


