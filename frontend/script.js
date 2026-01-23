/**
 * Movie Recommendation System - Frontend JavaScript
 * 
 * This file handles:
 * - API calls to the backend
 * - User interface interactions
 * - Displaying recommendations
 */

// ============================================
// CONFIGURATION
// ============================================

// Use environment variable or fallback to localhost
// For Vercel deployment, set API_BASE_URL in Environment Variables
const API_BASE_URL = process.env.API_BASE_URL || window.API_BASE_URL || 'http://127.0.0.1:8000';


// ============================================
// DOM ELEMENTS
// ============================================

// Tab buttons
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

// Movie recommendation elements
const movieInput = document.getElementById('movieInput');
const movieCount = document.getElementById('movieCount');
const movieSearchBtn = document.getElementById('movieSearchBtn');
const movieLoading = document.getElementById('movieLoading');
const movieError = document.getElementById('movieError');
const queryMovie = document.getElementById('queryMovie');
const movieResults = document.getElementById('movieResults');

// Taste Builder elements
const moviesToRate = document.getElementById('moviesToRate');
const getTasteRecommendations = document.getElementById('getTasteRecommendations');
const tasteLoading = document.getElementById('tasteLoading');
const tasteError = document.getElementById('tasteError');
const tasteStats = document.getElementById('tasteStats');
const tasteResults = document.getElementById('tasteResults');
const ratedCount = document.getElementById('ratedCount');
const progressFill = document.getElementById('progressFill');

// API status
const apiStatus = document.getElementById('apiStatus');


// ============================================
// TAB SWITCHING
// ============================================

tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Get the target tab
        const targetTab = button.getAttribute('data-tab');
        
        // Remove active class from all buttons and tabs
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked button and corresponding tab
        button.classList.add('active');
        document.getElementById(targetTab).classList.add('active');
    });
});


// ============================================
// API HEALTH CHECK
// ============================================

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            apiStatus.textContent = '✅ Online';
            apiStatus.style.color = '#4caf50';
        } else {
            apiStatus.textContent = '⚠️ Issues detected';
            apiStatus.style.color = '#ff9800';
        }
    } catch (error) {
        apiStatus.textContent = '❌ Offline';
        apiStatus.style.color = '#f44336';
        console.error('API health check failed:', error);
    }
}

// Check API health on page load
checkAPIHealth();


// ============================================
// MOVIE RECOMMENDATIONS
// ============================================

async function getMovieRecommendations() {
    const title = movieInput.value.trim();
    const k = parseInt(movieCount.value) || 5;
    
    // Validate input
    if (!title) {
        showError(movieError, 'Please enter a movie title');
        return;
    }
    
    // Reset UI
    hideError(movieError);
    hideElement(queryMovie);
    clearResults(movieResults);
    showLoading(movieLoading);
    
    try {
        // Call API
        const response = await fetch(
            `${API_BASE_URL}/recommend/movie?title=${encodeURIComponent(title)}&k=${k}`
        );
        
        const data = await response.json();
        
        hideLoading(movieLoading);
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }
        
        // Display query movie
        displayQueryMovie(data.query_movie);
        
        // Display recommendations
        displayMovieRecommendations(data.recommendations);
        
    } catch (error) {
        hideLoading(movieLoading);
        showError(movieError, error.message);
        console.error('Error:', error);
    }
}

function displayQueryMovie(movie) {
    queryMovie.innerHTML = `
        <h3>Selected Movie</h3>
        <div class="movie-info">
            <strong>Title:</strong> ${movie.title}<br>
            <strong>Genres:</strong> ${movie.genres || 'N/A'}<br>
            <strong>Overview:</strong> ${movie.overview}
        </div>
    `;
    showElement(queryMovie);
}

function displayMovieRecommendations(recommendations) {
    if (recommendations.length === 0) {
        movieResults.innerHTML = '<p style="text-align: center; color: #666;">No recommendations found.</p>';
        return;
    }
    
    movieResults.innerHTML = recommendations.map((movie, index) => `
        <div class="movie-card">
            <span class="movie-rank">#${index + 1}</span>
            <h3 class="movie-title">${movie.title}</h3>
            <div class="movie-genres">
                ${formatGenres(movie.genres)}
            </div>
            <p class="movie-overview">${truncateText(movie.overview, 150)}</p>
            <div class="movie-score">
                <span class="score-label">Similarity:</span>
                <span class="score-value">${(movie.similarity_score * 100).toFixed(1)}%</span>
            </div>
        </div>
    `).join('');
}


// ============================================
// USER RECOMMENDATIONS
// ============================================

async function getUserRecommendations() {
    const userId = parseInt(userInput.value);
    const k = parseInt(userCount.value) || 5;
    
    // Validate input
    if (!userId || userId < 1) {
        showError(userError, 'Please enter a valid user ID');
        return;
    }
    
    // Reset UI
    hideError(userError);
    hideElement(userStats);
    clearResults(userResults);
    showLoading(userLoading);
    
    try {
        // Call API
        const response = await fetch(
            `${API_BASE_URL}/recommend/user/${userId}?k=${k}`
        );
        
        const data = await response.json();
        
        hideLoading(userLoading);
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }
        
        // Display user stats
        displayUserStats(data.user_id, data.user_stats);
        
        // Display recommendations
        displayUserRecommendations(data.recommendations);
        
    } catch (error) {
        hideLoading(userLoading);
        showError(userError, error.message);
        console.error('Error:', error);
    }
}

function displayUserStats(userId, stats) {
    userStats.innerHTML = `
        <h3>User #${userId}</h3>
        <p><strong>Total Ratings:</strong> ${stats.total_ratings}</p>
        <p><strong>Liked Movies (≥4.0):</strong> ${stats.liked_movies}</p>
        <p style="margin-top: 10px; color: #667eea;">
            ✨ Based on this user's rating history, here are personalized recommendations:
        </p>
    `;
    showElement(userStats);
}

function displayUserRecommendations(recommendations) {
    if (recommendations.length === 0) {
        userResults.innerHTML = '<p style="text-align: center; color: #666;">No recommendations found. User may have rated all available movies.</p>';
        return;
    }
    
    userResults.innerHTML = recommendations.map((movie, index) => `
        <div class="movie-card">
            <span class="movie-rank">#${index + 1}</span>
            <h3 class="movie-title">${movie.title}</h3>
            <div class="movie-genres">
                ${formatGenres(movie.genres)}
            </div>
            <p class="movie-overview">${truncateText(movie.overview, 150)}</p>
            <div class="movie-score">
                <span class="score-label">Match Score:</span>
                <span class="score-value">${(movie.similarity_score * 100).toFixed(1)}%</span>
            </div>
        </div>
    `).join('');
}


// ============================================
// UTILITY FUNCTIONS
// ============================================

function showLoading(element) {
    element.classList.remove('hidden');
}

function hideLoading(element) {
    element.classList.add('hidden');
}

function showError(element, message) {
    element.textContent = `❌ ${message}`;
    element.classList.remove('hidden');
}

function hideError(element) {
    element.classList.add('hidden');
}

function showElement(element) {
    element.classList.remove('hidden');
}

function hideElement(element) {
    element.classList.add('hidden');
}

function clearResults(element) {
    element.innerHTML = '';
}

function formatGenres(genres) {
    if (!genres) return '<span class="genre-tag">Unknown</span>';
    
    return genres
        .split('|')
        .map(genre => `<span class="genre-tag">${genre.trim()}</span>`)
        .join('');
}

function truncateText(text, maxLength) {
    if (!text) return 'No description available.';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}


// ============================================
// EVENT LISTENERS
// ============================================

// Movie recommendations
movieSearchBtn.addEventListener('click', getMovieRecommendations);

movieInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        getMovieRecommendations();
    }
});

// Taste Builder
getTasteRecommendations.addEventListener('click', getTasteBasedRecommendations);


// ============================================
// TASTE BUILDER FUNCTIONALITY
// ============================================

// Popular movies to rate (diverse genres)
const popularMovies = [
    { movie_id: 1, title: "Toy Story", genres: "Adventure|Animation|Children|Comedy|Fantasy" },
    { movie_id: 2, title: "Jumanji", genres: "Adventure|Children|Fantasy" },
    { movie_id: 110, title: "Braveheart", genres: "Action|Drama|War" },
    { movie_id: 260, title: "Star Wars: Episode IV - A New Hope", genres: "Action|Adventure|Sci-Fi" },
    { movie_id: 296, title: "Pulp Fiction", genres: "Comedy|Crime|Drama|Thriller" },
    { movie_id: 318, title: "The Shawshank Redemption", genres: "Crime|Drama" },
    { movie_id: 356, title: "Forrest Gump", genres: "Comedy|Drama|Romance|War" },
    { movie_id: 480, title: "Jurassic Park", genres: "Action|Adventure|Sci-Fi|Thriller" },
    { movie_id: 527, title: "Schindler's List", genres: "Drama|War" },
    { movie_id: 589, title: "Terminator 2: Judgment Day", genres: "Action|Sci-Fi|Thriller" },
    { movie_id: 593, title: "The Silence of the Lambs", genres: "Crime|Horror|Thriller" },
    { movie_id: 1210, title: "Star Wars: Episode VI - Return of the Jedi", genres: "Action|Adventure|Sci-Fi" }
];

// Store user ratings
let userRatings = {};

function initializeTasteBuilder() {
    userRatings = {};
    moviesToRate.innerHTML = '';
    
    // Render movie rating cards
    popularMovies.forEach(movie => {
        const card = document.createElement('div');
        card.className = 'rate-movie-card';
        card.innerHTML = `
            <h4>${movie.title}</h4>
            <div class="rate-movie-genres">
                ${formatGenres(movie.genres)}
            </div>
            <div class="star-rating" data-movie-id="${movie.movie_id}">
                ${[1, 2, 3, 4, 5].map(rating => 
                    `<span class="star" data-rating="${rating}">★</span>`
                ).join('')}
            </div>
        `;
        moviesToRate.appendChild(card);
    });
    
    // Add star click listeners
    document.querySelectorAll('.star-rating').forEach(ratingDiv => {
        const movieId = parseInt(ratingDiv.dataset.movieId);
        const stars = ratingDiv.querySelectorAll('.star');
        
        stars.forEach(star => {
            star.addEventListener('click', () => {
                const rating = parseInt(star.dataset.rating);
                userRatings[movieId] = rating;
                
                // Update star display
                stars.forEach((s, index) => {
                    if (index < rating) {
                        s.classList.add('filled');
                    } else {
                        s.classList.remove('filled');
                    }
                });
                
                updateTasteProgress();
            });
            
            // Hover effect
            star.addEventListener('mouseenter', () => {
                const rating = parseInt(star.dataset.rating);
                stars.forEach((s, index) => {
                    if (index < rating) {
                        s.classList.add('active');
                    } else {
                        s.classList.remove('active');
                    }
                });
            });
        });
        
        ratingDiv.addEventListener('mouseleave', () => {
            stars.forEach(s => s.classList.remove('active'));
        });
    });
    
    updateTasteProgress();
}

function updateTasteProgress() {
    const count = Object.keys(userRatings).length;
    ratedCount.textContent = count;
    
    const progress = Math.min((count / 5) * 100, 100);
    progressFill.style.width = progress + '%';
    
    // Enable button if at least 5 movies rated
    getTasteRecommendations.disabled = count < 5;
    
    // Update button text
    if (count >= 5) {
        getTasteRecommendations.textContent = `Get My Personalized Recommendations (${count} movies rated)`;
    } else {
        getTasteRecommendations.textContent = `Rate at least ${5 - count} more movie${5 - count !== 1 ? 's' : ''}`;
    }
}

async function getTasteBasedRecommendations() {
    showLoading(tasteLoading);
    hideError(tasteError);
    tasteResults.innerHTML = '';
    tasteStats.classList.add('hidden');
    
    try {
        // Send ratings to backend
        const response = await fetch(`${API_BASE_URL}/recommend/taste-profile`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ratings: Object.entries(userRatings).map(([movie_id, rating]) => ({
                    movie_id: parseInt(movie_id),
                    rating: rating
                })),
                k: 10
            })
        });
        
        const data = await response.json();
        
        hideLoading(tasteLoading);
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }
        
        // Display stats
        displayTasteStats(data);
        
        // Display recommendations
        displayTasteRecommendations(data.recommendations);
        
    } catch (error) {
        hideLoading(tasteLoading);
        showError(tasteError, error.message);
    }
}

function displayTasteStats(data) {
    const likedMovies = popularMovies.filter(m => userRatings[m.movie_id] >= 4);
    
    tasteStats.innerHTML = `
        <h3>Your Taste Profile</h3>
        <p>You rated <strong>${data.ratings_used}</strong> movies, 
           loved <strong>${likedMovies.length}</strong> of them!</p>
        <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
            Based on your ratings, here are movies you'll love:
        </p>
    `;
    tasteStats.classList.remove('hidden');
}

function displayTasteRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        tasteResults.innerHTML = '<p class="no-results">No recommendations found. Try rating more movies!</p>';
        return;
    }
    
    tasteResults.innerHTML = recommendations.map((movie, index) => `
        <div class="movie-card">
            <span class="movie-rank">#${index + 1}</span>
            <h3 class="movie-title">${movie.title}</h3>
            <div class="movie-genres">
                ${formatGenres(movie.genres)}
            </div>
            <p class="movie-overview">${truncateText(movie.overview, 150)}</p>
            <div class="movie-score">
                <span class="score-label">Match:</span>
                <span class="score-value">${(movie.similarity_score * 100).toFixed(1)}%</span>
            </div>
        </div>
    `).join('');
}


// ============================================
// INITIALIZATION
// ============================================

// Initialize taste builder when user tab is clicked
tabButtons.forEach(button => {
    const originalListener = button.onclick;
    button.addEventListener('click', () => {
        if (button.dataset.tab === 'user-tab') {
            if (moviesToRate.children.length === 0) {
                initializeTasteBuilder();
            }
        }
    });
});

console.log('🎬 Movie Recommendation System loaded');
console.log('API URL:', API_BASE_URL);
