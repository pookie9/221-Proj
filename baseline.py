
import os
import random
try:
    from nltk import ngrams;
    import nltk
except:
    print "You need to install ntlk, do it with 'sudo pip install -U nltk'"
    exit()
if not os.path.isfile('moby_dick.txt'):
    os.system('wget https://www.gutenberg.org/files/2701/2701.txt; mv 2701.txt moby_dick.txt')

class LinearInterpolation:

    """
    N is the number of grams to use, e.g. N=3 means use unigrams, bigrams, and trigrams. 
    weights is how weight each the different n-gram models, if left to none it weights them all equally.
    """
    def __init__(self, N,weights=None):
        self.N=N
        if weights==None:
            self.weights=[1.0/self.N]*self.N
        est=lambda fdist,bins: LidstoneProbDist(fdist,0.2)
        self.models=[NgramModel(i+1) for i in range(self.N)]
    def train_sentence(self,sentence):
        for model in self.models:
            model.train_sentence(sentence)
    def prob_of_word(self,context,word):
        prob=0.0
        for weight,model in zip(self.weights,self.models):
            prob=prob+weight*model.prob_of_word(context,word)
        return prob

    def generate_word(self,context):
        r=random.random()        
        for word in self.models[0].vocab:
            for weight,model in zip(self.weights,self.models):
                r-=weight*model.prob_of_word(context,word)
            if r<=0.0:
                return word
        
    def generate_sentence(self):
        context=["<BEGIN>"]*self.N
        while context[-1]!="<END>":
            context.append(self.generate_word(context))
        while '<BEGIN>' in context:
            context.remove('<BEGIN>')
        while '<UNK>' in context:
            context.remove('<UNK>')
        return ' '.join(context[:-1])
        

class NgramModel:
    
    def __init__(self,N):
        self.N=N
        self.vocab={"<UNK>":0}#unknown word
        self.context_counts={}
        self.counts={}
        self.total_count=0

    def train_sentence(self,words):
        begin=["<BEGIN>"]*(self.N)
        begin.extend(words)
        begin.extend(["<END>"]*(self.N))
        words=begin
        for word in words:
            self.vocab[word]=True
        
        grams=ngrams(words,self.N)
        for gram in grams:
            gram=' '.join(gram)
            self.counts[gram]=self.counts.get(gram,0)+1
            self.total_count+=1
        if self.N>1:
            grams=ngrams(words,self.N-1)
            for gram in grams:
                gram=' '.join(gram)
                self.context_counts[gram]=self.context_counts.get(gram,0)+1

    def prob_of_word(self,context,word):
        if self.N>1:
            context=' '.join(context)
            return (self.counts.get(context+' '+word,0)+1.0)/(self.context_counts.get(context,0)+len(self.vocab))
        else:
            return (self.counts.get(word,0)+1.0)/(self.total_count+len(self.vocab))
    
    def generate_word(self,context):
        r=random.random()
        for word in self.vocab:
            r-=self.prob_of_word(context,word)            
            if r<0.0:
                return word
        
if __name__=='__main__':
#    model=LinearInterpolation(3)
    random.seed(10)
    model=LinearInterpolation(3)
    f=open('moby_dick.txt')
    text=f.read()
    sentences=nltk.sent_tokenize(text)
    for sentence in sentences:
        model.train_sentence(nltk.word_tokenize(sentence))
    
    gen_sentence=model.generate_sentence()
    print gen_sentence
    os.system('say "'+gen_sentence+'"')
