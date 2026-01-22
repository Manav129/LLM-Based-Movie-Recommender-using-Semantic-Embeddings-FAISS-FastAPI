# LLM-Based Movie Recommendation System - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [How It Works](#how-it-works)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Running the System](#running-the-system)
7. [API Endpoints](#api-endpoints)
8. [Understanding User IDs and Ratings](#understanding-user-ids-and-ratings)
9. [Technical Details](#technical-details)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is an **end-to-end movie recommendation system** that combines:
- **LLM-based embeddings** (using sentence-transformers)
- **Vector similarity search** (using FAISS)
- **Hybrid filtering** (content-based + collaborative)

### Key Features
- 🎬 **Content-based recommendations**: Find movies similar to a given movie
- 👤 **Personalized recommendations**: Get recommendations based on user rating history
- 🚀 **Fast similarity search**: Uses FAISS for efficient nearest-neighbor search
- 🌐 **REST API**: FastAPI backend with JSON responses
- 💻 **Web interface**: Simple HTML/CSS/JS frontend

### Technologies Used
- **Python 3.10+**
- **FastAPI 0.109.0** - Modern web framework for APIs
- **sentence-transformers 2.2.2** - For generating text embeddings
- **faiss-cpu 1.7.4** - Facebook's vector similarity search library
- **pandas 2.1.4** - Data manipulation
- **numpy 1.26.3** - Numerical computing
- **uvicorn** - ASGI server for FastAPI

---

## System Architecture

```
┌─────────────┐
│   Frontend  │  (HTML/CSS/JS)
│  (Browser)  │
└──────┬──────┘
       │ HTTP Requests
       ↓
┌─────────────────────────────────────────────────┐
│          FastAPI Backend (app/main.py)          │
│  - CORS middleware                              │
│  - API endpoints (/recommend/movie, /user)      │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────┐
│      Recommender Logic (src/recommender.py)     │
│  - Content-based filtering                      │
│  - Collaborative filtering                      │
│  - Hybrid recommendations                       │
└──────┬────────────────────────────┬─────────────┘
       │                            │
       ↓                            ↓
┌──────────────┐          ┌──────────────────┐
│  FAISS Index │          │  Movie Embeddings│
│  (artifacts/ │          │  (artifacts/     │
│   index.bin) │          │   embeddings.npy)│
└──────────────┘          └──────────────────┘
       │                            │
       └────────────┬───────────────┘
                    ↓
          ┌──────────────────┐
          │  MovieLens Data  │
          │  - movies.csv    │
          │  - ratings.csv   │
          │  (data/)         │
          └──────────────────┘
```

---

## How It Works

### 1. Content-Based Recommendations (By Movie)

**User asks**: "Show me movies similar to Inception"

**System workflow**:
1. Find "Inception" in the movie database
2. Get its embedding vector (384-dimensional)
3. Search FAISS index for similar vectors
4. Return top K most similar movies with scores

**Example**:
```bash
GET /recommend/movie?title=Inception&k=5
```

**Behind the scenes**:
- Each movie has an embedding created from: `title + genres + overview`
- FAISS uses **cosine similarity** to find similar vectors
- Distance scores are converted to similarity scores (0-1)

---

### 2. Personalized Recommendations (By User)

**User asks**: "Show me recommendations for User 123"

**System workflow**:
1. Get all ratings by User 123 from `ratings.csv`
2. Filter for highly-rated movies (rating >= 4.0)
3. Get embeddings for those liked movies
4. **Create user profile**: Average of all liked movie embeddings
5. Search FAISS for movies similar to this profile
6. Filter out movies the user already rated
7. Return top K recommendations

**Example**:
```bash
GET /recommend/user/123?k=10
```

**Why this works**:
- User's taste = Average of what they liked
- System finds movies similar to their overall taste
- Excludes movies they've already seen/rated

---

### 3. Understanding User IDs and Ratings

#### What is a User ID?
- **User IDs come from the MovieLens dataset** (`ratings.csv`)
- Each user has rated multiple movies with scores 0.5-5.0
- User IDs range from 1 to several thousand (depending on dataset size)

#### How Ratings Are Used
```python
# ratings.csv structure:
user_id,movie_id,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
1,6,4.0,964982224
```

**Rating scale**:
- **0.5 - 2.5**: Disliked
- **3.0 - 3.5**: Neutral/Mixed
- **4.0 - 5.0**: Liked (used for recommendations)

**Threshold**: The system uses `MIN_RATING_THRESHOLD = 4.0` (configurable in `src/config.py`)
- Only movies rated ≥ 4.0 are considered "liked"
- User profile is built from these liked movies

#### Why User IDs Matter
User-based recommendations **REQUIRE** rating history because:
1. System needs to know what the user likes
2. Creates a "taste profile" from liked movies
3. Finds new movies matching that taste
4. Excludes already-rated movies

**Without ratings**: User-based recommendations won't work (will throw error)

---

## Project Structure

```
LLM Based Recommendation System/
│
├── app/
│   ├── main.py              # FastAPI application (API server)
│   └── __pycache__/         # Python cache
│
├── src/
│   ├── config.py            # Configuration (paths, parameters)
│   ├── data_loader.py       # Load movies.csv and ratings.csv
│   ├── build_embeddings.py  # Generate embeddings using sentence-transformers
│   ├── build_faiss.py       # Build FAISS index for similarity search
│   ├── recommender.py       # Core recommendation logic
│   └── __pycache__/         # Python cache
│
├── scripts/
│   ├── download_movielens.py  # Download MovieLens dataset
│   └── run_pipeline.py        # Generate embeddings + build FAISS index
│
├── data/
│   ├── movies.csv           # Movie metadata (movie_id, title, genres, overview)
│   └── ratings.csv          # User ratings (user_id, movie_id, rating, timestamp)
│
├── artifacts/
│   ├── embeddings.npy       # Numpy array of movie embeddings
│   ├── index.bin            # FAISS index file
│   └── movies_with_embeddings.csv  # Movies + embeddings (for debugging)
│
├── frontend/
│   ├── index.html           # Web interface
│   ├── style.css            # Styling
│   └── script.js            # Frontend logic (API calls)
│
├── requirements.txt         # Python dependencies
├── README.md               # Quick start guide
├── COMPLETE_PROJECT_GUIDE.md  # Detailed step-by-step guide
├── TESTING_AND_DEPLOYMENT.md  # Testing and deployment instructions
└── PROJECT_DOCUMENTATION.md   # This file (comprehensive documentation)
```

### File Purposes

#### Core Application Files
- **`app/main.py`**: FastAPI server entry point. Defines all API endpoints, handles CORS, error handling, and startup initialization.

- **`src/recommender.py`**: Heart of the system. Contains:
  - `recommend_by_movie()` - Content-based recommendations
  - `recommend_for_user()` - Hybrid personalized recommendations
  - `get_user_profile()` - Build user taste profile from ratings

- **`src/build_embeddings.py`**: Generates embeddings for all movies using sentence-transformers model.

- **`src/build_faiss.py`**: Builds FAISS index for fast similarity search.

- **`src/data_loader.py`**: Loads movies.csv and ratings.csv into pandas DataFrames.

- **`src/config.py`**: Central configuration file for paths, model names, parameters.

#### Script Files
- **`scripts/download_movielens.py`**: Downloads MovieLens dataset from GroupLens website.

- **`scripts/run_pipeline.py`**: **Run this once** to:
  1. Generate embeddings for all movies
  2. Build FAISS index
  3. Save artifacts to disk

#### Data Files
- **`data/movies.csv`**: Movie metadata
  ```csv
  movie_id,title,genres,overview
  1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy,"Led by Woody..."
  ```

- **`data/ratings.csv`**: User rating history
  ```csv
  user_id,movie_id,rating,timestamp
  1,1,4.0,964982703
  ```

#### Generated Artifacts
- **`artifacts/embeddings.npy`**: NumPy array of shape `(num_movies, 384)` containing embeddings.

- **`artifacts/index.bin`**: FAISS index file for fast similarity search.

- **`artifacts/movies_with_embeddings.csv`**: Debug file showing movies with their embeddings.

---

## Installation & Setup

### Step 1: Clone/Download Project
```bash
cd "c:\Users\manav\OneDrive\Desktop\LLM Based Recommendation System"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

**Dependencies installed**:
- fastapi==0.109.0
- uvicorn[standard]==0.27.0
- pandas==2.1.4
- numpy==1.26.3
- sentence-transformers==2.2.2
- faiss-cpu==1.7.4
- requests==2.31.0

### Step 3: Download Dataset (if not present)
```powershell
python scripts/download_movielens.py
```

This downloads MovieLens dataset and extracts:
- `data/movies.csv`
- `data/ratings.csv`

### Step 4: Generate Embeddings and Build Index
```powershell
python scripts/run_pipeline.py
```

**What this does**:
1. Loads movies.csv
2. Generates embeddings using `all-MiniLM-L6-v2` model
3. Normalizes embeddings (for cosine similarity)
4. Builds FAISS index
5. Saves artifacts to `artifacts/` folder

**Expected output**:
```
======================================================================
MOVIE RECOMMENDATION SYSTEM - PIPELINE
======================================================================
📥 Loading data...
✅ Loaded 9,742 movies

🤖 Loading embedding model: all-MiniLM-L6-v2
✅ Model loaded

🔄 Generating embeddings...
100%|████████████████████████████████████| 9742/9742 [00:45<00:00, 215.47it/s]
✅ Generated 9,742 embeddings (384 dimensions)

💾 Saving embeddings...
✅ Embeddings saved

🏗️ Building FAISS index...
✅ FAISS index built (9,742 vectors)

💾 Saving FAISS index...
✅ FAISS index saved

======================================================================
✅ PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
```

**Time taken**: ~1-2 minutes (depending on hardware)

---

## Running the System

### Start the API Server

#### Option 1: Using Uvicorn (Recommended)
```powershell
uvicorn app.main:app --reload
```

#### Option 2: Using Python Directly
```powershell
python app/main.py
```

**Server starts on**: `http://127.0.0.1:8000`

**Expected output**:
```
======================================================================
🚀 STARTING MOVIE RECOMMENDATION API
======================================================================
🚀 Initializing recommender system...
✅ FAISS index loaded: 9742 vectors, dimension=384
✅ Embeddings loaded: (9742, 384)
✅ Loaded 9742 movies
✅ Loaded 100836 ratings from 610 users
✅ Recommender initialized!
   Movies: 9742
   Ratings: 100836
   Users: 610

======================================================================
✅ API SERVER READY!
======================================================================

📡 Available endpoints:
   GET  /health                      - Health check
   GET  /recommend/movie             - Content-based recommendations
   GET  /recommend/user/{user_id}    - User-based recommendations
   GET  /search                      - Search movies
   GET  /movie/{movie_id}            - Get movie details

📚 Documentation:
   Swagger UI: http://127.0.0.1:8000/docs
   ReDoc:      http://127.0.0.1:8000/redoc
======================================================================

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Access the Frontend
Open browser and navigate to:
```
file:///c:/Users/manav/OneDrive/Desktop/LLM%20Based%20Recommendation%20System/frontend/index.html
```

Or use a local server:
```powershell
cd frontend
python -m http.server 8080
```
Then open: `http://localhost:8080`

---

## API Endpoints

### 1. Health Check
**Endpoint**: `GET /health`

**Purpose**: Verify API is running and data is loaded

**Example**:
```bash
curl http://127.0.0.1:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "Movie Recommendation API",
  "movies_loaded": 9742,
  "ratings_loaded": 100836
}
```

---

### 2. Movie Recommendations (Content-Based)
**Endpoint**: `GET /recommend/movie`

**Parameters**:
- `title` (required): Movie title (partial match works)
- `k` (optional): Number of recommendations (default: 10, max: 50)

**Example**:
```bash
curl "http://127.0.0.1:8000/recommend/movie?title=Inception&k=5"
```

**Response**:
```json
{
  "success": true,
  "method": "content-based",
  "query_movie": {
    "movie_id": 79132,
    "title": "Inception (2010)",
    "genres": "Action|Crime|Drama|Mystery|Sci-Fi|Thriller",
    "overview": "Cobb, a skilled thief..."
  },
  "recommendations": [
    {
      "movie_id": 89745,
      "title": "Shutter Island (2010)",
      "genres": "Drama|Mystery|Thriller",
      "overview": "World War II soldier-turned-U.S. Marshal...",
      "similarity_score": 0.876
    },
    ...
  ],
  "count": 5
}
```

---

### 3. User Recommendations (Personalized)
**Endpoint**: `GET /recommend/user/{user_id}`

**Parameters**:
- `user_id` (required): User identifier (path parameter)
- `k` (optional): Number of recommendations (default: 10, max: 50)

**Example**:
```bash
curl "http://127.0.0.1:8000/recommend/user/123?k=10"
```

**Response**:
```json
{
  "success": true,
  "method": "hybrid",
  "user_id": 123,
  "user_stats": {
    "total_ratings": 45,
    "liked_movies": 28
  },
  "recommendations": [
    {
      "movie_id": 1234,
      "title": "The Matrix (1999)",
      "genres": "Action|Sci-Fi|Thriller",
      "overview": "Set in the 22nd century...",
      "similarity_score": 0.912
    },
    ...
  ],
  "count": 10
}
```

**Error Cases**:
- **404**: User not found or has no ratings
- **404**: User has no ratings >= 4.0 (threshold)

---

### 4. Search Movies
**Endpoint**: `GET /search`

**Parameters**:
- `q` (required): Search query
- `limit` (optional): Max results (default: 10, max: 50)

**Example**:
```bash
curl "http://127.0.0.1:8000/search?q=dark&limit=5"
```

**Response**:
```json
{
  "success": true,
  "query": "dark",
  "results": [
    {
      "movie_id": 155,
      "title": "The Dark Knight (2008)",
      "genres": "Action|Crime|Drama|Thriller"
    },
    ...
  ],
  "count": 5
}
```

---

### 5. Get Movie Details
**Endpoint**: `GET /movie/{movie_id}`

**Parameters**:
- `movie_id` (required): Movie identifier (path parameter)

**Example**:
```bash
curl http://127.0.0.1:8000/movie/155
```

**Response**:
```json
{
  "success": true,
  "movie": {
    "movie_id": 155,
    "title": "The Dark Knight (2008)",
    "genres": "Action|Crime|Drama|Thriller",
    "overview": "Batman raises the stakes..."
  }
}
```

---

## Understanding User IDs and Ratings

### How to Find Valid User IDs

#### Method 1: Check ratings.csv
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/ratings.csv'); print(f'Users: {df['user_id'].min()}-{df['user_id'].max()}'); print(f'Total users: {df['user_id'].nunique()}')"
```

**Output**:
```
Users: 1-610
Total users: 610
```

#### Method 2: Use Python Interactive
```python
import pandas as pd

# Load ratings
ratings = pd.read_csv('data/ratings.csv')

# Get all unique user IDs
user_ids = ratings['user_id'].unique()
print(f"Valid user IDs: {user_ids[:10]}...")  # First 10

# Get user with most ratings
user_counts = ratings['user_id'].value_counts()
top_user = user_counts.index[0]
print(f"User {top_user} has {user_counts.iloc[0]} ratings")
```

### Understanding User Rating History

```python
import pandas as pd

ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Get ratings for User 1
user_1_ratings = ratings[ratings['user_id'] == 1]

# Merge with movie titles
user_1_movies = user_1_ratings.merge(movies, on='movie_id')

# Sort by rating (highest first)
user_1_movies = user_1_movies.sort_values('rating', ascending=False)

print(f"User 1 has rated {len(user_1_movies)} movies")
print("\nTop 5 rated movies:")
print(user_1_movies[['title', 'rating', 'genres']].head())
```

### Rating Distribution
```python
import pandas as pd

ratings = pd.read_csv('data/ratings.csv')

# Count ratings by score
rating_counts = ratings['rating'].value_counts().sort_index()
print("Rating distribution:")
print(rating_counts)

# Percentage of high ratings (>= 4.0)
high_ratings_pct = (ratings['rating'] >= 4.0).mean() * 100
print(f"\n{high_ratings_pct:.1f}% of ratings are >= 4.0")
```

---

## Technical Details

### Embedding Generation

**Model**: `all-MiniLM-L6-v2` (from sentence-transformers)
- **Dimensions**: 384
- **Max sequence length**: 256 tokens
- **Speed**: ~215 movies/second on CPU

**Text processed**:
```python
text = f"{title}. {genres}. {overview}"
```

**Example**:
```
Input: "Inception (2010). Action|Sci-Fi|Thriller. A skilled thief..."
Output: [0.023, -0.145, 0.678, ..., -0.234]  # 384 dimensions
```

**Normalization**:
```python
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```
- Normalizes vectors to unit length
- Enables cosine similarity = dot product

---

### FAISS Index

**Type**: `IndexFlatIP` (Flat Index with Inner Product)
- **Inner Product** = Cosine similarity for normalized vectors
- **Exact search** (not approximate)
- **Fast**: ~0.1ms for single query on 10K movies

**Why FAISS?**
- Optimized for large-scale similarity search
- Supports GPU acceleration (if using faiss-gpu)
- Industry standard (used by Facebook, Google, etc.)

**Search complexity**: O(n) for flat index, but highly optimized

---

### Similarity Scores

**Conversion from distance to similarity**:
```python
# FAISS returns L2 distance for normalized vectors
# For normalized vectors: distance^2 = 2 * (1 - cosine_similarity)
similarity = 1 - (distance ** 2) / 2
```

**Score interpretation**:
- **0.9 - 1.0**: Very similar (same genre, similar plot)
- **0.7 - 0.9**: Similar (related themes, overlapping genres)
- **0.5 - 0.7**: Somewhat similar
- **< 0.5**: Not very similar

---

### Collaborative Filtering (User Profile)

**How user profile is created**:

1. Get user's highly-rated movies (rating >= 4.0)
2. Get embeddings for those movies
3. Compute average embedding:
   ```python
   user_profile = np.mean(liked_embeddings, axis=0)
   ```

**Why averaging works**:
- Captures user's general taste
- Balances across different genres/themes
- Similar to "centroid" of user's preferences

**Example**:
```
User likes:
- The Matrix (Sci-Fi, Action) → [0.1, 0.9, 0.3, ...]
- Inception (Sci-Fi, Thriller) → [0.2, 0.8, 0.4, ...]
- Interstellar (Sci-Fi, Drama) → [0.15, 0.85, 0.35, ...]

User Profile = Average:
→ [0.15, 0.85, 0.35, ...]  # "Sci-Fi-ish" vector
```

System then finds movies similar to this profile.

---

### Configuration Parameters

**File**: `src/config.py`

**Key parameters**:
```python
# Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Recommendations
DEFAULT_K = 10              # Default number of recommendations
MIN_RATING_THRESHOLD = 4.0  # Minimum rating to consider "liked"

# Paths
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
```

**Tunable parameters**:
- **MIN_RATING_THRESHOLD**: Lower = more movies in user profile (less picky)
- **DEFAULT_K**: Higher = more recommendations returned
- **EMBEDDING_MODEL**: Can use different models (e.g., "all-mpnet-base-v2" for better quality, slower)

---

## Troubleshooting

### Issue 1: "FileNotFoundError: artifacts not found"

**Cause**: Pipeline hasn't been run yet

**Solution**:
```powershell
python scripts/run_pipeline.py
```

---

### Issue 2: "User has no ratings above 4.0"

**Cause**: User exists but hasn't rated any movies highly

**Solution**: Try a different user ID or lower `MIN_RATING_THRESHOLD` in `src/config.py`

---

### Issue 3: "ImportError: No module named 'huggingface_hub'"

**Cause**: Missing dependency for sentence-transformers

**Solution**:
```powershell
pip install huggingface-hub
```

---

### Issue 4: CORS errors in frontend

**Cause**: Browser blocking cross-origin requests

**Solution**: Ensure `CORS_ORIGINS` in `src/config.py` includes your frontend origin:
```python
CORS_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "null"  # For file:// protocol
]
```

---

### Issue 5: "Movie not found"

**Cause**: Movie title doesn't match exactly

**Solution**: Use partial match or search endpoint:
```bash
curl "http://127.0.0.1:8000/search?q=dark"
```

---

### Issue 6: Slow embedding generation

**Cause**: Running on CPU, large dataset

**Solutions**:
- Use smaller dataset (MovieLens 100K instead of 1M/10M)
- Use GPU: Install `faiss-gpu` and `torch` with CUDA
- Use faster model: Change to smaller embedding model

---

## Summary of Commands Run

### Initial Setup
```powershell
# Navigate to project directory
cd "c:\Users\manav\OneDrive\Desktop\LLM Based Recommendation System"

# Install dependencies
pip install -r requirements.txt

# Download dataset (if needed)
python scripts/download_movielens.py
```

### Generate Artifacts (Run Once)
```powershell
# Generate embeddings and build FAISS index
python scripts/run_pipeline.py
```

### Start API Server
```powershell
# Start FastAPI server with auto-reload
uvicorn app.main:app --reload

# Alternative: Run directly
python app/main.py
```

### Testing Endpoints
```powershell
# Health check
curl http://127.0.0.1:8000/health

# Movie recommendations
curl "http://127.0.0.1:8000/recommend/movie?title=Inception&k=5"

# User recommendations
curl "http://127.0.0.1:8000/recommend/user/123?k=10"

# Search movies
curl "http://127.0.0.1:8000/search?q=dark&limit=5"

# Get movie details
curl http://127.0.0.1:8000/movie/155
```

### Access Documentation
```
Swagger UI: http://127.0.0.1:8000/docs
ReDoc: http://127.0.0.1:8000/redoc
```

---

## Conclusion

This recommendation system demonstrates a **production-ready hybrid approach**:
- ✅ Fast similarity search with FAISS
- ✅ LLM-based embeddings for semantic understanding
- ✅ Hybrid filtering (content + collaborative)
- ✅ RESTful API with FastAPI
- ✅ Simple web interface

**Key strengths**:
- Handles cold-start (can recommend by movie without user history)
- Personalized recommendations using rating history
- Fast inference (<100ms per request)
- Scalable architecture

**Potential improvements**:
- Add matrix factorization (SVD) for better collaborative filtering
- Implement caching for frequently-requested recommendations
- Add user authentication and personalized profiles
- Deploy to cloud (AWS, GCP, Azure) with Docker
- Add A/B testing framework

---

**Documentation created**: 2026-01-21  
**Last updated**: 2026-01-21  
**Author**: Movie Recommendation System Team  
**Version**: 1.0.0
