import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

def getVocabList():
    #GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
    #cell array of the words
    #   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
    #   and returns a cell array of the words in vocabList.
    
    
    ## Read the fixed vocabulary list
    fd = open('vocab.txt', 'r')
    
    # Store all dictionary words in cell array vocab{}
    n = 1899  # Total number of words in the dictionary
    
    # For ease of implementation, we use a struct to map the strings => integers
    # In practice, you'll want to use some form of hashmap
    vocabList = {}

    for i in range(n):
        # Word Index (can ignore since it will be = i)
        line = fd.readline()
        elements = line.split('\t')
        # Actual Word
        word = elements[1].strip()     #끝에 달려있는 개행문자 빼주기.
        vocabList[i] = word 
    
    fd.close()
    return vocabList
