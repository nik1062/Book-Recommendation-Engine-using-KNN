import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load dataset
df_books = pd.read_csv("BX-Books.csv", sep=";", encoding="latin-1", on_bad_lines='skip', low_memory=False)
df_ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", encoding="latin-1", on_bad_lines='skip', low_memory=False)
df_users = pd.read_csv("BX-Users.csv", sep=";", encoding="latin-1", on_bad_lines='skip', low_memory=False)

# Rename columns
df_books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']
df_ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
df_users.columns = ['User-ID', 'Location', 'Age']

# Merge books and ratings dataset
df = df_ratings.merge(df_books, on='ISBN')

# Filter users with at least 200 ratings
user_counts = df_ratings['User-ID'].value_counts()
df = df[df['User-ID'].isin(user_counts[user_counts >= 200].index)]

# Filter books with at least 100 ratings
book_counts = df['Book-Title'].value_counts()
df = df[df['Book-Title'].isin(book_counts[book_counts >= 100].index)]

# Create pivot table
book_pivot = df.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)

# Fit Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_pivot)

# Function to get recommendations
def get_recommends(book_title):
    if book_title not in book_pivot.index:
        return "Book not found in dataset."
    
    book_index = book_pivot.index.get_loc(book_title)
    distances, indices = model.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)
    
    recommended_books = []
    for i in range(1, len(indices[0])):
        recommended_books.append([book_pivot.index[indices[0][i]], distances[0][i]])
    
    return [book_title, recommended_books]

# Test the function
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
