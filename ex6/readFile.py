import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

def readFile(filename):
    #READFILE reads a file and returns its entire contents 
    #   file_contents = READFILE(filename) reads a file and returns its entire
    #   contents in file_contents
    #
    
    # Load File
    try:
        fd = open(filename, 'r')
        file_contents = fd.read()
        fd.close()
    except FileNotFoundError:
        file_contents = ''
        print('Unable to open ', filename, '\n')
        
    return file_contents


