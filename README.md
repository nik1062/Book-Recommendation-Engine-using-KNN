# Book-Recommendation-Engine-using-KNN
Dataset Preparation

Uses the Book-Crossings dataset (1.1M ratings, 270K books, 90K users).
Filters users with 200+ ratings and books with 100+ ratings for accuracy.
Creating a User-Book Matrix

Converts ratings into a pivot table where rows = books, columns = users.
Fills missing ratings with 0 (unrated books).

Using K-Nearest Neighbors (KNN)

Applies cosine similarity to measure how closely books are related.
Finds the 5 most similar books for any given title.

Generating Recommendations

Input: "The Queen of the Damned (Vampire Chronicles (Paperback))"
Output:
python
Copy
Edit
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.7939], 
    ['The Witching Hour', 0.7448], 
    ['Interview with the Vampire', 0.7345],
    ['The Tale of the Body Thief', 0.5376],
    ['The Vampire Lestat', 0.5178]
  ]
]
