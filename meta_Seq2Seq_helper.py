def get_unique_words(sentences):
    # Get a list of all the unique words in a list of sentences
    #
    # Input
    #  sentences: list of sentence strings
    # Output
    #   words : list of all unique words in sentences
    words = []
    for s in sentences:
        for w in s.split(' '):  # words
            if w not in words:
                words.append(w)
    return words
    
    
def make_hashable(G):
    # Separate and sort stings, to make unique string identifier for an episode
    #
    # Input
    #   G : string of elements separate by \n, specifying the structure of an episode
    G_str = str(G).split('\n')
    G_str.sort()
    out = '\n'.join(G_str)
    return out.strip()

def pad_seq(seq, max_length):
    # Pad sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def build_padded_var(list_seq, lang):
    # Transform python list to a padded torch tensor
    #
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of symbols)
    #  lang : language object for translation into indices
    #
    # Output
    #  z_padded : LongTensor (n x max_length)
    #  z_lengths : python list of sequence lengths (list of scalars)
    n = len(list_seq)
    if n == 0:
        return [], []
    z_eos = [z+[EOS_token] for z in list_seq]
    z_lengths = [len(z) for z in z_eos]
    max_len = max(z_lengths)
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [tf.expand_dims(lang.variableFromSymbols(z, add_eos=False), 0) for z in z_padded]
    z_padded = tf.concat(z_padded, axis=0)
    return z_padded, z_lengths
    
def tabu_update(tabu_list, identifier):
    # Add all elements of "identifier" to the 'tabu_list', and return updated list
    if isinstance(identifier, (list, set)):
        tabu_list = tabu_list.union(identifier)
    elif isinstance(identifier, str):
        tabu_list.add(identifier)
    else:
        assert False
    return tabu_list
    
class Lang:
    # Class for converting strings/words to numerical indices, and vice versa.
    #  Should use separate class for input language (English) and output language (actions)
    #
    def __init__(self, symbols):
        # symbols : list of all possible symbols
        n = len(symbols)
        self.symbols = symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token}
        self.symbol2index = {SOS_token: n, EOS_token: n+1}
        for idx, s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)

    def variableFromSymbols(self, mylist, add_eos=True):
        # Convert a list of symbols to a tensor of indices (adding a EOS token at end)
        #
        # Input
        #  mylist : list of m symbols
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] indices of each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos:
            mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = tf.Variable(indices, dtype=tf.int64)
        return output

    def symbolsFromVector(self, v):
        # Convert indices to symbols, breaking where we get a EOS token
        #
        # Input
        #  v : list of m indices
        #
        # Output
        #  mylist : list of m or m-1 symbols (excluding EOS)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist
