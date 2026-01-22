"""
FastAPI Backend for Movie Recommendation System

This is the main API server that exposes recommendation endpoints.
The frontend (HTML/CSS/JS) will call these endpoints to get recommendations.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

# Add src directory to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.recommender import (
    initialize_recommender,
    recommend_by_movie,
    recommend_for_user,
    recommend_by_taste_profile,
    search_movies,
    get_movie_by_id
)
from src.config import CORS_ORIGINS, DEFAULT_K


# ============================================
# REQUEST MODELS
# ============================================

class MovieRating(BaseModel):
    movie_id: int
    rating: float

class TasteProfileRequest(BaseModel):
    ratings: List[MovieRating]
    k: int = 10


# ============================================
# CREATE FASTAPI APP
# ============================================

app = FastAPI(
    title="Movie Recommendation API",
    description="LLM-based hybrid movie recommendation system using embeddings and FAISS",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc"  # ReDoc documentation
)


# ============================================
# CORS MIDDLEWARE
# ============================================

# CORS (Cross-Origin Resource Sharing) allows the frontend to call the backend
# even if they're hosted on different domains/ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Which origins can access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    This runs once when the server starts.
    We initialize the recommender here to load all data into memory.
    This way, recommendations are fast (no loading on each request).
    """
    print("Starting API server...")
    try:
        initialize_recommender()
        print("API server ready.")
        print("   GET  /recommend/movie             - Content-based recommendations")
        print("   GET  /recommend/user/{user_id}    - User-based recommendations")
        print("   GET  /search                      - Search movies")
        print("   GET  /movie/{movie_id}            - Get movie details")
        print("\n📚 Documentation:")
        print("   Swagger UI: http://127.0.0.1:8000/docs")
        print("   ReDoc:      http://127.0.0.1:8000/redoc")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print("\n" + "=" * 70)
        print("❌ STARTUP FAILED")
        print("=" * 70)
        print(f"\n{e}")
        print("\n💡 Solution: Run the pipeline first")
        print("   python scripts/run_pipeline.py")
        print("\n" + "=" * 70 + "\n")
        # Don't exit - let FastAPI start but endpoints will return errors


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """
    Root endpoint - Welcome message.
    """
    return {
        "message": "Welcome to Movie Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend_by_movie": "/recommend/movie?title=YourMovie&k=10",
            "recommend_for_user": "/recommend/user/{user_id}?k=10",
            "search": "/search?q=query&limit=10",
            "movie_details": "/movie/{movie_id}"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Use this to verify the API is running and ready.
    
    Returns:
        dict: Status information
    """
    try:
        from src.recommender import get_loaded_data
        _, _, movies_df, ratings_df = get_loaded_data()
        
        return {
            "status": "healthy",
            "service": "Movie Recommendation API",
            "movies_loaded": len(movies_df),
            "ratings_loaded": len(ratings_df) if ratings_df is not None else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/recommend/movie")
async def get_movie_recommendations(
    title: str = Query(..., description="Movie title to find similar movies for"),
    k: int = Query(DEFAULT_K, ge=1, le=50, description="Number of recommendations")
):
    """
    Get content-based recommendations for a movie.
    
    This endpoint finds movies similar to the given movie based on
    content features (title, genres, plot).
    
    Args:
        title: Movie title (partial match works)
        k: Number of recommendations (default: 10, max: 50)
    
    Returns:
        dict: Query movie info and list of recommendations
    
    Example:
        GET /recommend/movie?title=Interstellar&k=5
    """
    try:
        result = recommend_by_movie(title, k=k)
        return {
            "success": True,
            "method": "content-based",
            "query_movie": result['query_movie'],
            "recommendations": result['recommendations'],
            "count": len(result['recommendations'])
        }
    
    except ValueError as e:
        # Movie not found
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/recommend/user/{user_id}")
async def get_user_recommendations(
    user_id: int,
    k: int = Query(DEFAULT_K, ge=1, le=50, description="Number of recommendations")
):
    """
    Get personalized recommendations for a user.
    
    This endpoint generates recommendations based on the user's
    rating history (hybrid collaborative + content-based filtering).
    
    Args:
        user_id: User identifier
        k: Number of recommendations (default: 10, max: 50)
    
    Returns:
        dict: User info and list of personalized recommendations
    
    Example:
        GET /recommend/user/123?k=10
    """
    try:
        result = recommend_for_user(user_id, k=k)
        return {
            "success": True,
            "method": "hybrid",
            "user_id": result['user_id'],
            "user_stats": {
                "total_ratings": result['num_ratings'],
                "liked_movies": result['num_liked']
            },
            "recommendations": result['recommendations'],
            "count": len(result['recommendations'])
        }
    
    except ValueError as e:
        # User not found or insufficient data
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/recommend/taste-profile")
async def get_taste_profile_recommendations(request: TasteProfileRequest):
    """
    Get personalized recommendations based on user's taste profile.
    
    This endpoint allows users to rate movies on-the-spot and get
    instant personalized recommendations without needing a user account.
    
    Args:
        request: TasteProfileRequest with ratings list and k value
    
    Returns:
        dict: Personalized recommendations based on provided ratings
    
    Example:
        POST /recommend/taste-profile
        {
            "ratings": [
                {"movie_id": 1, "rating": 5.0},
                {"movie_id": 260, "rating": 4.5},
                {"movie_id": 318, "rating": 5.0}
            ],
            "k": 10
        }
    """
    try:
        # Convert ratings to dictionary
        ratings_dict = {r.movie_id: r.rating for r in request.ratings}
        
        # Get recommendations
        result = recommend_by_taste_profile(ratings_dict, k=request.k)
        
        return {
            "success": True,
            "method": "taste-profile",
            "ratings_used": len(ratings_dict),
            "recommendations": result['recommendations'],
            "count": len(result['recommendations'])
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/search")
async def search_movies_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """
    Search for movies by title.
    
    This endpoint performs a case-insensitive partial match on movie titles.
    Useful for autocomplete/search features in the frontend.
    
    Args:
        q: Search query string
        limit: Maximum results to return (default: 10, max: 50)
    
    Returns:
        dict: List of matching movies
    
    Example:
        GET /search?q=dark&limit=5
    """
    try:
        results = search_movies(q, limit=limit)
        return {
            "success": True,
            "query": q,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/movie/{movie_id}")
async def get_movie_details(movie_id: int):
    """
    Get details for a specific movie by ID.
    
    Args:
        movie_id: Movie identifier
    
    Returns:
        dict: Movie details
    
    Example:
        GET /movie/123
    """
    try:
        movie = get_movie_by_id(movie_id)
        return {
            "success": True,
            "movie": movie
        }
    
    except ValueError as e:
        # Movie not found
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    Custom 404 error handler.
    """
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not Found",
            "detail": str(exc.detail) if hasattr(exc, 'detail') else "Resource not found"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Custom 500 error handler.
    """
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": str(exc.detail) if hasattr(exc, 'detail') else "An unexpected error occurred"
        }
    )


# ============================================
# RUN THE SERVER
# ============================================

if __name__ == "__main__":
    """
    Run the server directly with: python app/main.py
    
    However, it's better to use uvicorn:
    uvicorn app.main:app --reload
    
    Options:
    --reload: Auto-reload on code changes (development)
    --host 0.0.0.0: Listen on all network interfaces
    --port 8000: Port number (default: 8000)
    """
    import uvicorn
    
    print("\n🚀 Starting development server...")
    print("💡 TIP: Use 'uvicorn app.main:app --reload' for auto-reload\n")
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
