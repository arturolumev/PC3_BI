from flask import Flask, render_template, request, make_response, g
from redis import Redis
from math import sqrt
import os
import socket
import random
import json
import logging

import csv
from math import sqrt
from builtins import zip

option_a = os.getenv('OPTION_A', "Persona 1")
option_b = os.getenv('OPTION_B', "Persona 2")
hostname = socket.gethostname()

app = Flask(__name__)

users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0, "Norah Jones": 4.5, "Phoenix": 5.0, "Slightly Stoopid": 1.5, "The Strokes": 2.5, "Vampire Weekend": 2.0},
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5, "Deadmau5": 4.0, "Phoenix": 2.0, "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0, "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5, "Slightly Stoopid": 1.0},
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0, "Deadmau5": 4.5, "Phoenix": 3.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 2.0},
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0, "Norah Jones": 4.0, "The Strokes": 4.0, "Vampire Weekend": 1.0},
         "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0, "Phoenix": 5.0, "Slightly Stoopid": 4.5, "The Strokes": 4.0, "Vampire Weekend": 4.0},
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0, "Norah Jones": 3.0, "Phoenix": 5.0, "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0, "Phoenix": 4.0, "Slightly Stoopid": 2.5, "The Strokes": 3.0}
        }

gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

def get_redis():
    if not hasattr(g, 'redis'):
        g.redis = Redis(host="redis", db=0, socket_timeout=5)
    return g.redis

################################

def manhattan(rating1, rating2):
    """Computes the Manhattan distance. Both rating1 and rating2 are dictionaries
       of the form {'The Strokes': 3.0, 'Slightly Stoopid': 2.5}"""
    distance = 0
    commonRatings = False 
    for key in rating1:
        if key in rating2:
            distance += abs(rating1[key] - rating2[key])
            commonRatings = True
    if commonRatings:
        return distance
    else:
        return -1 #Indicates no ratings in common


def computeNearestNeighbor(username, users):
    """creates a sorted list of users based on their distance to username"""
    distances = []
    for user in users:
        if user != username:
            distance = manhattan(users[user], users[username])
            distances.append((distance, user))
    # sort based on distance -- closest first
    distances.sort()
    return distances

def recommend(username, users):
    """Give list of recommendations"""
    # first find nearest neighbor
    nearest = computeNearestNeighbor(username, users)[0][1]

    recommendations = []
    # now find bands neighbor rated that user didn't
    neighborRatings = users[nearest]
    userRatings = users[username]
    for artist in neighborRatings:
        if not artist in userRatings:
            recommendations.append((artist, neighborRatings[artist]))
    # using the fn sorted for variety - sort is more efficient
    return sorted(recommendations, key=lambda artistTuple: artistTuple[1], reverse = True)

#print( recommend('Hailey', users))
result = recommend('Hailey', users)

################################################################

################################################################
# USANDO CSV

def load_users_from_csv(filename):
    users = {}
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            userId = row['userId']
            movieId = row['movieId']
            rating = float(row['rating'])
            
            if userId not in users:
                users[userId] = {}
            users[userId][movieId] = rating

    return users

# Cargar los usuarios desde el archivo CSV
users2 = load_users_from_csv('ratings.csv')

def pearson_correlation(rating1, rating2):
    """Computes the Pearson correlation. Both rating1 and rating2 are dictionaries
       of the form {'The Strokes': 3.0, 'Slightly Stoopid': 2.5}"""
    numerator = 0
    sum_rating1 = 0
    sum_rating2 = 0
    sum_sq_rating1 = 0
    sum_sq_rating2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            numerator += (rating1[key] - sum(rating1.values()) / n) * (rating2[key] - sum(rating2.values()) / n)
            sum_rating1 += rating1[key]
            sum_rating2 += rating2[key]
            sum_sq_rating1 += rating1[key] ** 2
            sum_sq_rating2 += rating2[key] ** 2
    denominator = sqrt((sum_sq_rating1 - sum_rating1 ** 2 / n) * (sum_sq_rating2 - sum_rating2 ** 2 / n))
    if not denominator:
        return 0.0
    else:
        return numerator / denominator

def computeNearestNeighbor(username, users):
    """creates a sorted list of users based on their distance to username"""
    distances = []
    for user in users:
        if user != username:
            distance = pearson_correlation(users[user], users[username])
            distances.append((distance, user))
    # sort based on distance -- closest first
    distances.sort(reverse=True)
    return distances

def recommend(username, users):
    """Give list of recommendations"""
    # first find nearest neighbor
    nearest = computeNearestNeighbor(username, users)[1:6]

    print(f"Los 5 vecinos más cercanos a {username} son:")
    for neighbor in nearest:
        print(neighbor[1])

# Ejemplo de uso
#print(users)
print(pearson_correlation(users2['1'], users2['5']))

################################################################

@app.route("/", methods=['POST','GET'])
def distancias():
    #distancia_manhattan = manhattan(users['Angelica'], users["Bill"])
    distancia_pearson = pearson_correlation(users2['1'], users2['5'])
    voter_id = request.cookies.get('voter_id')
    if not voter_id:
        voter_id = hex(random.getrandbits(64))[2:-1]
    vote = None
    if request.method == 'POST':
        redis = get_redis()
        user_1 = request.form['option_a']
        user_2 = request.form['option_b']
        distancia_manhattan = str(manhattan(users[user_1], users[user_2])) 
        #distancia_pearson = str(pearson(users[user_1], users[user_2]))
        data = json.dumps({'voter_id': voter_id,'distancia_manhattan': distancia_manhattan, 'distancia_pearson': distancia_pearson})
        #data = json.dumps({'voter_id': voter_id, 'distancia_manhattan': distancia_manhattan})
        redis.rpush('distancias', data)
    resp = make_response(render_template(
        'index.html',
        option_a=option_a,
        option_b=option_b,
        hostname=hostname,
    ))
    resp.set_cookie('voter_id', voter_id)
    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)
