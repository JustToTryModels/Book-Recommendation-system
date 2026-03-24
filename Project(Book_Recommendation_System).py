import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_and_prepare_data():
    # Load your final filtered dataframe
    final_filtered_df = pd.read_csv('final_filtered_df.csv')

    # Load the dataframe containing book URLs
    book_urls_df = pd.read_csv("Books.csv")
    book_urls_df.rename(columns={'Book-Title': 'title'}, inplace=True)

    # Merge the dataframes on the title
    final_filtered_df = final_filtered_df.merge(book_urls_df, on='title', how='left')

    # URL to replace
    url1 = 'http://images.amazon.com/images/P/0690040784.01.LZZZZZZZ.jpg'
    url2 = 'http://images.amazon.com/images/P/0451172817.01.LZZZZZZZ.jpg'
    url3 = 'http://images.amazon.com/images/P/0312084986.01.LZZZZZZZ.jpg'
    url4 = 'http://images.amazon.com/images/P/1590400356.01.LZZZZZZZ.jpg'

    # Replace URL based on condition
    final_filtered_df.loc[final_filtered_df['title'] == 'Jacob Have I Loved', 'Image-URL-L'] = url1
    final_filtered_df.loc[final_filtered_df['title'] == 'Needful Things', 'Image-URL-L'] = url2
    final_filtered_df.loc[final_filtered_df['title'] == 'All Creatures Great and Small', 'Image-URL-L'] = url3
    final_filtered_df.loc[final_filtered_df['title'] == "The Kitchen God's Wife", 'Image-URL-L'] = url4

    # Create the book-user matrix
    book_user_mat = final_filtered_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    # Calculate the cosine similarity matrix
    cosine_sim = cosine_similarity(book_user_mat)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=book_user_mat.index, columns=book_user_mat.index)

    return final_filtered_df, cosine_sim_df

final_filtered_df, cosine_sim_df = load_and_prepare_data()

def get_top_similar_books(book_title, n=10):
    if book_title not in cosine_sim_df.index:
        return "Book not found in the database."
    
    similar_scores = cosine_sim_df[book_title]
    similar_books = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_books

# Streamlit app
st.title('Book Recommendation System')

# Updated CSS: Using Caslon for text and keeping horizontal title scrolling
st.markdown("""
    <style>
    /* Applying Caslon to the main content */
    html, body, [class*="css"], [class*="st-"], .stMarkdown, p, div, span, label {
        font-family: 'Caslon', 'Libre Caslon Text', 'Big Caslon', 'Book Antiqua', 'Georgia', serif !important;
    }
    
    /* Keep buttons and inputs Sans-Serif for better UI readability */
    button, input, select, textarea {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }

    .subheader {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1a73e8;
    }
    
    .stButton > button {
        font-size: 16px;
        background-color: #4CAF50;
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Horizontal Scroll for long book titles */
    .scroll-title {
        display: block;
        font-size: 18px;
        font-weight: bold;
        white-space: nowrap;
        overflow-x: auto;
        padding-bottom: 8px;
        margin-bottom: 5px;
    }
    
    /* Scrollbar styling for Chrome/Safari/Edge */
    .scroll-title::-webkit-scrollbar {
        height: 5px;
    }
    .scroll-title::-webkit-scrollbar-thumb {
        background: #bbb;
        border-radius: 10px;
    }

    .author-info {
        margin-top: 5px;
        font-size: 14px;
        font-style: italic;
        color: #444;
    }
    
    .year-info {
        font-size: 12px;
        margin-top: 3px;
        margin-left: 10px;
        color: #777;
    }

    img {
        object-fit: contain;
        max-height: 300px;
        width: auto;
        display: block;
        margin: 0 auto;
    }

    hr {
        border: none !important;
        border-top: 6px solid #222 !important;
        margin: 30px 0 !important;
        opacity: 1 !important;
    }

    .book-column {
        position: relative;
        padding: 20px;
        border: 1px solid #eee;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        transition: transform 0.2s ease;
    }
    
    .book-column:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }

    .extra-space {
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='subheader'>Let Us Help You Choose Your Next Book!</p>", unsafe_allow_html=True)
st.image('https://img.freepik.com/premium-vector/bookcase-with-books_182089-197.jpg', use_container_width=True)

all_books = final_filtered_df['title'].unique().tolist()
book_title = st.selectbox('Enter a book title:', all_books, index=None, placeholder="Choose or enter a book title...", key='book_title')

num_recommendations = st.number_input('Enter the number of recommendations:', min_value=1, max_value=50, value=10)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommended_book' not in st.session_state:
    st.session_state.recommended_book = None
if 'recommended_num' not in st.session_state:
    st.session_state.recommended_num = None

if st.button('Recommend books'):
    if book_title:
        similar_books = get_top_similar_books(book_title, num_recommendations)
        st.session_state.recommendations = similar_books
        st.session_state.recommended_book = book_title
        st.session_state.recommended_num = num_recommendations
    else:
        st.session_state.recommendations = None
        st.warning("Please select a book first.")

if st.session_state.recommendations is not None:
    similar_books = st.session_state.recommendations
    rec_book = st.session_state.recommended_book
    rec_num = st.session_state.recommended_num

    if isinstance(similar_books, str):
        st.error(similar_books)
    else:
        st.markdown(f"<div style='font-size:16px; margin-bottom:15px;'>Top {rec_num} recommendations for '<strong>{rec_book}</strong>':</div>", unsafe_allow_html=True)
        
        for i in range(0, len(similar_books), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(similar_books):
                    book = similar_books.index[i + j]
                    book_info = final_filtered_df[final_filtered_df['title'] == book].iloc[0]
                    with cols[j]:
                        st.markdown(f"""
                        <div class='book-column'>
                            <div class='book-info'>
                                <div class='scroll-title'>{i + j + 1}. {book}</div>
                                <div class='author-info'>by {book_info['Book-Author']}</div>
                                <div class='year-info'>{book_info['Year-Of-Publication']}</div>
                            </div>
                            <img src='{book_info['Image-URL-L']}'>
                        </div>
                        """, unsafe_allow_html=True)
            if i < len(similar_books) - 3:
                st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("<div class='extra-space'></div>", unsafe_allow_html=True)
        st.image('https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true', use_container_width=True)
