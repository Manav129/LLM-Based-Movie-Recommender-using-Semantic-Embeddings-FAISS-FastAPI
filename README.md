# 🎬 Movie Recommendation System

A complete movie recommendation system using machine learning embeddings and vector similarity search. This project demonstrates how modern recommendation engines work by combining content-based filtering with collaborative filtering approaches.

## ✨ What This Project Does

The system provides two ways to discover movies:

**🔍 Find Similar Movies** - Search for any movie and get recommendations based on similar content, genres, and plot themes. Uses natural language processing to understand what makes movies similar beyond just matching keywords.

**⭐ Build Your Taste Profile** - Rate a few movies you know, and the system instantly generates personalized recommendations. No account needed, works right away with just 5 ratings. This is similar to how Netflix or Spotify understand your preferences.

The system currently works with **9,708 movies** from the MovieLens dataset and can return recommendations in under **100ms**.

## How It Works

The core technology uses sentence embeddings from pre-trained language models. Each movie description gets converted into a 384-dimensional vector that captures its semantic meaning. When you search for similar movies or build a taste profile, the system uses FAISS (Facebook AI Similarity Search) to find the closest matches in this vector space.

For the taste profile feature, your ratings get averaged into a single vector representing your preferences, then the system finds movies near that point in the embedding space. This approach handles complex tastes naturally - if you like both action and romance, it will recommend movies that blend both genres.

## Tech Stack

**Backend**
- Python 3.10
- FastAPI for the REST API
- Uvicorn as the ASGI server
- SentenceTransformers for generating embeddings
- FAISS for fast similarity search
- pandas and numpy for data processing

**Frontend**
- Plain HTML, CSS, and JavaScript
- No frameworks, just vanilla JS
- Responsive design that works on mobile

**Data**
- MovieLens Latest Small dataset (9,708 movies, 100,836 ratings)
- Can be easily swapped with other datasets

## Project Structure

```
.
├── data/                   # Raw CSV files
│   ├── movies.csv          # Movie information
│   └── ratings.csv         # User ratings
├── artifacts/              # Pre-computed files (generated)
│   ├── movie_embeddings.npy
│   ├── movie_metadata.pkl
│   └── faiss_index.index
├── src/                    # Core logic
│   ├── config.py           # Settings and paths
│   ├── data_loader.py      # Load and clean data
│   ├── build_embeddings.py # Generate embeddings
│   ├── build_faiss.py      # Build search index
│   └── recommender.py      # Recommendation algorithms
├── scripts/
│   ├── run_pipeline.py     # Build all artifacts
│   └── download_movielens.py
├── app/
│   └── main.py             # FastAPI application
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── requirements.txt
```

## Getting Started

**Prerequisites**
- Python 3.10 or higher
- About 500MB free disk space for dependencies and models
- A web browser

**Installation**

1. Clone this repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Download the MovieLens dataset (optional, included by default)
```bash
python scripts/download_movielens.py
```

4. Build the recommendation system (this takes 2-5 minutes on first run)
```bash
python scripts/run_pipeline.py
```

This step generates embeddings for all movies and builds the FAISS index. It only needs to run once unless you change the dataset.

5. Start the backend API server
```bash
uvicorn app.main:app --reload --port 8000
```

The API server will start at http://127.0.0.1:8000

6. Start the frontend (in a new terminal)
```bash
cd frontend
python -m http.server 9000
```

Then open http://localhost:9000 in your browser

**Quick Start (Both servers)**
```bash
# Terminal 1 - Backend
uvicorn app.main:app --reload

# Terminal 2 - Frontend
cd frontend && python -m http.server 9000
```

## Using the System

**Content-Based Search**

Type any movie title in the search box (try "Inception" or "The Dark Knight") and get instant recommendations for similar movies. The system understands semantic similarity, so it finds movies with similar themes, tones, and genres rather than just matching keywords.

**Taste Profile Builder**

Switch to the "Build Your Taste Profile" tab and rate at least 5 movies from the selection. The system analyzes your ratings and finds movies that match your preferences. The more movies you rate, the better the recommendations get, but 5-10 ratings usually work well.

Try rating movies across different genres to see how the system handles diverse tastes. It's pretty good at finding movies that blend your preferences.

## API Documentation

The FastAPI server includes auto-generated documentation:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

**Main Endpoints**

`GET /health`
Check if the server is running and how many movies are loaded.

`GET /recommend/movie`
Get similar movies based on a title.
- Parameters: `title` (string), `k` (int, default 10)
- Example: `/recommend/movie?title=Inception&k=5`

`POST /recommend/taste-profile`
Get personalized recommendations from a list of ratings.
- Body: `{"ratings": [{"movie_id": 1, "rating": 5.0}], "k": 10}`

`GET /search`
Search for movies by title (useful for autocomplete).
- Parameters: `q` (string), `limit` (int, default 10)

## Data Format

If you want to use your own dataset, the CSVs need these columns:

**movies.csv**
- `movie_id` - Unique integer identifier
- `title` - Movie title (string)
- `genres` - Pipe-separated genres like "Action|Thriller|Sci-Fi"
- `overview` - Plot description or summary (string)

**ratings.csv**
- `user_id` - Integer user ID
- `movie_id` - Integer movie ID (must match movies.csv)
- `rating` - Float rating value (typically 0.5 to 5.0)

After updating the data files, re-run `python scripts/run_pipeline.py` to rebuild the artifacts.

## 🚀 Performance

The system is designed to be fast:
- Embedding generation: ~45 seconds for 9,708 movies (one-time)
- FAISS index building: <1 second
- Recommendation query: <100ms
- Memory usage: ~200MB with everything loaded

## 🎯 Features

✅ Content-based movie recommendations  
✅ Personalized taste profile builder  
✅ Fast vector similarity search with FAISS  
✅ Clean, responsive UI  
✅ RESTful API with auto-generated docs  
✅ No external dependencies for frontend  
✅ Works completely offline once set up

## 🐛 Troubleshooting

**Port already in use**
```bash
# Kill process on port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

**CORS errors**
Make sure the frontend is running on an HTTP server (port 9000), not opening index.html directly as `file://`

**Module not found**
```bash
pip install -r requirements.txt --upgrade
```

## 📝 License

MIT License - feel free to use this for learning or build something on top of it.

## Acknowledgments

- MovieLens dataset from GroupLens Research
- SentenceTransformers by UKPLab
- FAISS by Facebook AI Research
- Built as a learning project to understand modern recommendation systems
