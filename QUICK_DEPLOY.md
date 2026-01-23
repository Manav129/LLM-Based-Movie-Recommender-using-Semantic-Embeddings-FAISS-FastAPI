# 🚀 Quick Deployment Guide

## Current Status
- ✅ **Frontend**: Deployed on Vercel (static site)
- ⏳ **Backend**: Needs to be deployed on Render

## Step-by-Step Deployment

### 1️⃣ Deploy Backend to Render

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Click "New +" → "Web Service"**

3. **Connect your GitHub repository**

4. **Configure the service:**
   - **Name**: `moviematch-backend` (or any name you prefer)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install --upgrade pip setuptools && pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT app.main:app
     ```
   - **Plan**: Free

5. **Add Environment Variables** (optional):
   - `PYTHON_VERSION`: `3.11`

6. **Click "Create Web Service"**

7. **Wait for deployment** (5-10 minutes for first deploy)
   - Render will install dependencies and start your API
   - Watch the logs for any errors

8. **Copy your Render URL** when deployment succeeds:
   - Example: `https://moviematch-backend-abcd.onrender.com`

### 2️⃣ Connect Frontend to Backend

1. **Open `frontend/index.html`** in your code editor

2. **Find line 10** and replace the placeholder:
   ```javascript
   // Change this line:
   window.API_BASE_URL = 'REPLACE_WITH_YOUR_RENDER_URL';
   
   // To your actual Render URL:
   window.API_BASE_URL = 'https://moviematch-backend-abcd.onrender.com';
   ```

3. **Commit and push changes:**
   ```bash
   git add frontend/index.html
   git commit -m "Configure production API URL"
   git push
   ```

4. **Vercel will auto-deploy** the updated frontend

### 3️⃣ Test Your Application

1. **Open your Vercel URL**
   - Example: `https://your-app.vercel.app`

2. **Check API Status** in the top-right corner:
   - Should show "✅ Online" in green

3. **Test a movie search:**
   - Try: "The Dark Knight" or "Inception"

4. **Test Taste Builder:**
   - Rate 5 movies and get recommendations

## 🐛 Troubleshooting

### Backend Issues

**Problem**: Render deployment fails
- Check the logs in Render dashboard
- Common issues:
  - Memory error: Use `gunicorn -w 1` (single worker)
  - Missing dependencies: Verify requirements.txt

**Problem**: API shows "❌ Offline"
- Verify Render service is running (not sleeping)
- Check the Render URL in browser: `https://your-backend.onrender.com/health`
- Should return: `{"status":"healthy"}`

**Problem**: Render service goes to sleep (Free plan)
- Free tier services sleep after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds to wake up
- Solution: Use a service like [UptimeRobot](https://uptimerobot.com/) to ping every 14 minutes

### Frontend Issues

**Problem**: "Failed to fetch"
- Verify API_BASE_URL in index.html is correct
- Check browser console for CORS errors
- Verify backend is running on Render

**Problem**: CORS errors
- Backend already has CORS configured for all origins
- If issues persist, check Render logs

### Common Mistakes

1. ❌ Forgot to update API_BASE_URL in index.html
2. ❌ Render URL has trailing slash (remove it)
3. ❌ Backend service stopped/sleeping on Render Free tier
4. ❌ Wrong Render URL (check Render dashboard for correct URL)

## 📊 Expected Behavior

### First Load
1. Frontend loads instantly (Vercel)
2. API health check runs
3. Status shows "✅ Online" if backend is awake
4. If backend is sleeping (Free tier), first request takes 30-60s

### After Backend Wakes Up
- All requests fast (<1s)
- Recommendations work smoothly
- Backend stays awake for 15 minutes

## 🔧 Local Development

To run locally:

```bash
# Terminal 1: Start backend
cd "c:\Users\manav\OneDrive\Desktop\LLM Based Recommendation System"
uvicorn app.main:app --reload

# Terminal 2: Start frontend (optional - can just open index.html)
cd frontend
python -m http.server 8080
```

Then open: `http://localhost:8080`

## 📝 URLs Summary

| Service | URL |
|---------|-----|
| Frontend (Vercel) | Update with your Vercel URL |
| Backend (Render) | Update with your Render URL |
| GitHub Repo | https://github.com/Manav129/llm-based-movie-recommender1 |

---

**Need help?** Check the logs:
- Vercel logs: Project → Deployments → Click deployment → Logs
- Render logs: Dashboard → Your service → Logs tab
