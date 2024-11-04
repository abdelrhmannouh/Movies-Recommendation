import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import google.generativeai as genai

# Configure the Google Generative AI API
genai.configure(api_key="AIzaSyCPQTU8iOnu6u3ftlV35NjNYGeixuM7Gu8")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load datasets
data_encoded_deploy = pd.read_csv('data_encoded_deploy.csv')
x_deploy = pd.read_csv('x_deploy.csv')

# Initialize KNN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(x_deploy)

# Define options for genres, lengths, and year categories
genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
          'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
          'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']

lengths = ['Short Film', 'Long Film', 'Medium Film']
year_categories = ['Classic', 'Modern', 'Old']

# Set background image and adjust layout using custom CSS
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images.pexels.com/photos/7991486/pexels-photo-7991486.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-size: cover;
    background-position: center;
    color: white;
}
header, .css-1v0mbdj {
    display: none; /* Hides header and sidebar */
}
h1, h2, h3, h4, h5, h6 {
    color: white;
    text-shadow: none; /* Remove text shadows */
}
label, p {
    color: white;
}
.stSelectbox, .stMultiselect {
    background-color: rgba(255, 255, 255, 0.7);
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    color: black;
}
.stButton>button {
    background-color: #ff6347;
    color: white;
    padding: 10px;
    border-radius: 10px;
}
.stMarkdown {
    background-color: rgba(0, 0, 0, 0.6);
    color: white;
    padding: 15px;
    border-radius: 10px;
}
.movie-card {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 10px;
    text-align: left;
    width: 80%;
    margin: 10px auto;
}
.movie-title {
    font-size: 1.4em;
    font-weight: bold;
    color: black;
}
.movie-details {
    font-size: 1.1em;
    color: black;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# App title and header
st.title('🎬 Movie Recommendation System')
st.header('🔍 Select Your Movie Preferences')

# Input widgets for user selections
selected_genres = st.multiselect('🎥 Genres', genres, format_func=lambda x: x)
selected_length = st.selectbox('⏳ Film Length', lengths)
selected_year_category = st.selectbox('🗓️ Year Category', year_categories)

# Create the input dictionary for user selection
column_dict = {genre: (1 if genre in selected_genres else 0) for genre in genres}
column_dict.update({
    'year_category_Classic': (1 if selected_year_category == 'Classic' else 0),
    'year_category_Modern': (1 if selected_year_category == 'Modern' else 0),
    'year_category_Old': (1 if selected_year_category == 'Old' else 0),
    'time of film_Short Film': (1 if selected_length == 'Short Film' else 0),
    'time of film_long Film': (1 if selected_length == 'Long Film' else 0),
    'time of film_med time': (1 if selected_length == 'Medium Film' else 0),
})

# Convert input to DataFrame for KNN model
sample_df = pd.DataFrame([column_dict])

# Get movie recommendations using KNN
distances, indices = knn.kneighbors(sample_df)
nearest_neighbor_points = data_encoded_deploy.iloc[indices.flatten()]

# Display movie recommendations
st.header('🍿 Recommended Movies 🍿')

for index, row in nearest_neighbor_points.iterrows():
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">🎬 {row['title']}</div>
        <div class="movie-details">📅 Release Year: {row['release_year']}</div>
        <div class="movie-details">⭐ Rating: {row['rating']}/10</div>
    </div>
    """, unsafe_allow_html=True)

# User description for AI recommendation
st.header('🤖 Describe the Film You Want')
film_description = st.text_area("Enter a description of the film you're looking for:")

# Additional recommendations using Google Generative AI
if st.button('Get AI Recommendations'):
    if film_description.strip():  # Check if the user provided a description
        try:
            # Use the user's description to generate a recommendation
            response = model.generate_content(
                f"Recommend a film based on the following description: {film_description}. Please avoid explicit content."
            )
            # Check for valid response
            if response and hasattr(response, 'text') and response.text:
                st.subheader('🎬 AI Recommended Film')
                st.markdown(f"**Recommended Film:** {response.text}")
            else:
                st.error("No valid recommendation returned. Please try a different description.")

        except Exception as e:
            st.error(f"Error occurred: {e}")
    else:
        st.error("Please enter a description of the film you are looking for.")
