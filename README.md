# cs224n  
Stanford deep nlp course  

Orignial course materials are in:   
http://web.stanford.edu/class/cs224n/  


Answers to CS224n problem sets, practice exams; only contains written and coding answers, original problem sets please refer to Stanford Official course site.  
This is for personal interest, you are welcom to use the material but keep in mind this is not guaranteed to be the right answer.  
It is also really appreciated if you could point out anything wrong or provide any suggestions on improvement of my answer.  


Errors in PS:  
1, PS1.2, in q2_neural.py function sanity_check, last parameter should be 'data', not 'params'.   
gradcheck_naive(lambda params:    
    forward_backward_prop(data, labels, params, dimensions), data)  


Contents:  
tfwarmup: Tensorflow environment preparation    
In this folder I trained 3 language models: VanillaRNN/LSTM/GRU, to get my tensorflow work properly.  
The dataset is from @dennybritz's rnn tutorial: https://github.com/dennybritz/rnn-tutorial-rnnlm.   
He's also got this blog http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/, easy to understand.  

Assignment 1:  
1, Softmax    
2, NN basics (forward/backward propogation, Cross Entropy, Softmax and Sigmoid)    
3, word2vec (Cross Entropy loss, Negative sampling, Skipgram vs CBOW)    
4, Sentiment Analysis    


Ongoing topics:  
Word2Vec, CBOW, GloVe  
RNN, LSTM, GRU  
Dependency Parser, TreeRNN  
Language Model,   
Machine Translation,  
Question Answering  