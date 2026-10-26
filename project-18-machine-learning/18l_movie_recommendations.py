
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Movie ratings from different users
# 0 means the user has not rated the movie

ratings = pd.DataFrame({
    "Space Mission": [5, 5, 1, 1, 4],
    "Galaxy Adventure": [5, 4, 1, 2, 5],
    "Alien Planet": [5, 0, 1, 0, 4],
    "Love in Paris": [1, 1, 5, 5, 0],
    "Wedding Day": [0, 0, 5, 4, 1],
    "Haunted House": [0, 0, 4, 5, 0]
}, index=["Alice", "Bob", "Charlie", "David", "Emma"])

# 2. Calculate similarity between users
similarity = cosine_similarity(ratings)

similarity_df = pd.DataFrame(
    similarity,
    index=ratings.index,
    columns=ratings.index
)

print("User Similarity:")
print(similarity_df.round(2))


# 3. Recommend movies based on similar users
def recommend(user_name):
    # Find users similar to our selected user
    similar_users = similarity_df[user_name].drop(user_name)

    # Sort from most similar to least similar
    similar_users = similar_users.sort_values(ascending=False)

    # Movies the selected user has not rated
    unwatched = ratings.loc[user_name]
    unwatched = unwatched[unwatched == 0].index

    recommendations = {}

    for movie in unwatched:
        total_score = 0
        total_similarity = 0

        for other_user, similarity_score in similar_users.items():
            rating = ratings.loc[other_user, movie]

            if rating > 0 and similarity_score > 0:
                total_score += rating * similarity_score
                total_similarity += similarity_score

        if total_similarity > 0:
            recommendations[movie] = total_score / total_similarity

    recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nRecommendations for {user_name}:")

    for movie, score in recommendations:
        print(f"{movie} (Predicted rating: {score:.2f}/5)")


# 4. Test our recommendation system
recommend("Bob")
recommend("Alice")

