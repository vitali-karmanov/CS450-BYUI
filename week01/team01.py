# -*- coding: utf-8 -*-
"""
@author: Vitali Karmanov

"""
import random
import numpy as np
import decimal

class Movie:

    def __init__(self, title="", year=0, runtime=0):
        self.title = title
        self.year = year
        self.runtime = runtime
        
        if runtime < 0:
            self.runtime = 0
     
    def __repr__(self):
        return self.title + " (" + str(self.year) + ") " + "- " + str(self.runtime)
        
    def formatRuntime(self):
        hours = self.runtime // 60
        minutes = self.runtime % 60
        return(hours, minutes)
        
def create_movie_list():
    listMovies = []
    movie1 = Movie("Avengers 1", 2015, 90)
    movie2 = Movie("Avengers 2", 2016, 110)
    movie3 = Movie("Avengers 3", 2017, 120)
    movie4 = Movie("Avengers 4", 2018, 170)
    movie5 = Movie("Avengers 5", 2019, 80)
    
    listMovies.append(movie1)
    listMovies.append(movie2)
    listMovies.append(movie3)
    listMovies.append(movie4)
    listMovies.append(movie5)
    
    return (listMovies)
    
def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 1

        # Lets have the views range from 100-10000
        views = random.randint(100, 200)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array
    
def main():

    movieList = create_movie_list()
    
    for movie in movieList:
           print (movie)
            
    newList = [movie for movie in movieList if movie.runtime > 150]
        
    for movie in newList:
        print (movie)
        
    ratings = {}
    for movie in movieList:
        rating = float(decimal.Decimal(random.randrange(100, 500))/100)
        ratings[movie.title] = rating
        
    print (ratings)

    data = get_movie_data()

    print(data)
    
    rows = data.shape[0]
    cols = data.shape[1]

    print("There are {} rows and {} cols".format(rows, cols))
    
    
    print(data[0:2])
    

    print(data[:,-2:])

    
    print(data[:,1])



if __name__== "__main__":
  main()