from flask import Flask, render_template, request, make_response, g
from redis import Redis
from math import sqrt
import os
import socket
import random
import json
import logging

#### PC2 DAEA
import csv
from math import sqrt
from builtins import zip

#### PC 3 BI
import fileinput
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cityblock

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

################################################################
# EVALUACION 3 BI

# Ruta al archivo que quieres modificar
archivo = '/ml-10M100K/ratings.dat'

# Iterar sobre cada línea del archivo
with fileinput.FileInput(archivo, inplace=True, backup='.bak') as file:
    for line in file:
        # Reemplazar '::' por '\t' en cada línea
        print(line.replace('::', '\t'), end='')
        
# Convert MovieLens data to binary using numpy_to_binary function
def movie_lens_to_binary(input_file, output_file):
    # Load MovieLens data using Pandas
    ratings = pd.read_csv(input_file, sep='\t', header=None,
                          names=['userId', 'movieId', 'rating', 'rating_timestamp'])
    # Convert to NumPy array
    np_data = np.array(ratings[['userId', 'movieId', 'rating']])
    # Write to binary file
    with open(output_file, "wb") as bin_file:
        bin_file.write(np_data.astype(np.int32).tobytes())
movie_lens_to_binary('/content/ml-10M100K/ratings.dat', 'output_binary.bin')

def binary_to_pandas(bin_file, num_rows=10):
    # Read binary data into NumPy array
    with open(bin_file, 'rb') as f:
        binary_data = f.read()

    # Convert binary data back to NumPy array
    np_data = np.frombuffer(binary_data, dtype=np.int32).reshape(-1, 3)  # Assuming 3 columns

    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(np_data, columns=['userId', 'movieId', 'rating'])

    # Display the equivalent of ratings.head(10)
    print(df.head(num_rows))

# Usage
binary_to_pandas('output_binary.bin', num_rows=10)

def binary_to_pandas_with_stats(bin_file, num_rows=10):
    # Read binary data into NumPy array
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    # Convert binary data back to NumPy array
    np_data = np.frombuffer(binary_data, dtype=np.int32).reshape(-1, 3)  # Assuming 3 columns
    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(np_data, columns=['userId', 'movieId', 'rating'])
    # Calculate max and min values for 'userId'
    userId_max = df['userId'].max()
    userId_min = df['userId'].min()
    num_rows_df = len(df.index)
    return userId_max, userId_min, num_rows_df
# Usage
userId_max, userId_min, num_rows_df = binary_to_pandas_with_stats('output_binary.bin', num_rows=10)

print(f"Maximum userId: {userId_max}")
print(f"Minimum userId: {userId_min}")
print(f"Number of rows: {num_rows_df}")

#16 seg
import numpy as np
import pandas as pd

def binary_to_pandas_with_stats(bin_file, num_rows=10):
    # Read binary data into NumPy array
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    # Convert binary data back to NumPy array
    np_data = np.frombuffer(binary_data, dtype=np.int32).reshape(-1, 3)  # Assuming 3 columns
    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(np_data, columns=['userId', 'movieId', 'rating'])
    return df
def consolidate_data(df):
    # Group by 'userId' and 'movieId' and calculate the mean of 'rating'
    consolidated_df = df.groupby(['userId', 'movieId'])['rating'].mean().unstack()
    return consolidated_df
df = binary_to_pandas_with_stats('output_binary.bin', num_rows=10)

# Consolidate data
consolidated_df = consolidate_data(df)
print("Consolidated data:")
print(consolidated_df)

#it takes 32 seconds
#comparate


def computeNearestNeighbor(dataframe, target_user, distance_metric=cityblock):
    distances = np.zeros(len(dataframe))  # Initialize a NumPy array
    # Iterate over each row (user) in the DataFrame
    for i, (index, row) in enumerate(dataframe.iterrows()):
        if index == target_user:
            continue  # Skip the target user itself
        # Calculate the distance between the target user and the current user
        distance = distance_metric(dataframe.loc[target_user].fillna(0), row.fillna(0))
        distances[i] = distance
    # Get the indices that would sort the array, and then sort the distances accordingly
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    return list(zip(dataframe.index[sorted_indices], sorted_distances))
# Example usage
# Assuming your DataFrame is named 'ratings_df'
target_user_id = 1
neighbors = computeNearestNeighbor(consolidated_df, target_user_id)
# Print the nearest neighbors and their distances

print(neighbors[0][0])


################################################################

@app.route("/", methods=['POST','GET'])
def distancias():
    #distancia_manhattan = manhattan(users['Angelica'], users["Bill"])
    distancia_pearson = pearson_correlation(users2['1'], users2['5'])
    pc3 = neighbors[0][0]
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
        data = json.dumps({'voter_id': voter_id,'distancia_manhattan': distancia_manhattan, 'distancia_pearson': distancia_pearson, 'pc3': pc3})
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
