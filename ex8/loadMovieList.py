def loadMovieList():
    #GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    #cell array of the words
    #   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
    #   and returns a cell array of the words in movieList.
    
    ## Read the fixed movieulary list
    fid = open('movie_ids.txt', 'r')    # txt file 의 인코딩을 ANSI 로 지정.
    
    # Store all movies in cell array movie{}
    n = 1682  # Total number of movies 
    
    movieList = {}
    for i in range(n):
        # Read line
        line = fid.readline()
        # Word Index (can ignore since it will be = i)
        movieName = line.split(' ', 1)[1] # split(a, b) : a 를 기준으로 b 번만 분리
        # Actual Word
        movieList[i] = movieName.strip()

    fid.close()

    return movieList
