# Deployment Guide - MovieMatch

This guide covers deploying MovieMatch to **Render** (backend) and **Vercel** (frontend).

---

## Part 1: Deploy Backend to Render

### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Authorize GitHub access

### Step 2: Create Web Service
1. Click **New** → **Web Service**
2. Connect your GitHub repo: `LLM-Based-Movie-Recommender-using-Semantic-Embeddings-FAISS-FastAPI`
3. Configure:
   - **Name**: `moviematch-backend`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free tier (or paid for better performance)

### Step 3: Set Environment Variables (Optional)
In Render dashboard:
1. Go to your service → **Environment**
2. Add if needed:
   ```
   FRONTEND_URL=https://your-vercel-app.vercel.app
   ```

### Step 4: Deploy
- Render auto-deploys on push to main
- Wait for build completion (first deploy takes ~5-10 mins)
- Your backend URL: `https://moviematch-backend.onrender.com`

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Create Vercel Account
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub

### Step 2: Import GitHub Project
1. Click **Add New Project** → **Import Git Repository**
2. Select your GitHub repo
3. Configure:
   - **Framework**: Other (static)
   - **Output Directory**: `frontend`
   - **Build Command**: Leave empty

### Step 3: Environment Variables
1. Go to **Settings** → **Environment Variables**
2. Add:
   ```
   NEXT_PUBLIC_API_BASE_URL=https://moviematch-backend.onrender.com
   ```

### Step 4: Deploy
- Click **Deploy**
- Your frontend URL: `https://your-vercel-app.vercel.app`

---

## Step 3: Connect Frontend to Backend

### Update Frontend API URL
In your deployed frontend (Vercel), the backend URL is injected via:

**Option A: Create `frontend/config.js`**
```javascript
window.API_BASE_URL = 'https://moviematch-backend.onrender.com';
```

Then add before `</head>` in `index.html`:
```html
<script src="config.js"></script>
```

**Option B: Edit `frontend/index.html`** directly and add before closing `</body>`:
```html
<script>
  window.API_BASE_URL = 'https://moviematch-backend.onrender.com';
</script>
```

---

## Troubleshooting

### Backend Not Connecting
- **Check CORS**: Make sure Render backend allows requests from your Vercel domain
- **Check URL**: Use HTTPS in production
- **Logs**: View logs in Render dashboard

### Embeddings Not Loading
- **Cold start**: First request takes longer as embeddings load
- **Memory**: Free tier on Render has limited memory (512MB)
- **Solution**: Upgrade to paid tier or use smaller model

### API Timeouts
- **Increase timeout** in Vercel (Settings → Functions)
- **Optimize backend**: Consider using faster similarity search

---

## Deployment Checklist

- [ ] GitHub repo created and code pushed
- [ ] Render account created
- [ ] Backend deployed to Render
- [ ] Vercel account created  
- [ ] Frontend deployed to Vercel
- [ ] Frontend API URL updated to Render backend
- [ ] Test recommendations from deployed frontend
- [ ] Add project to resume with deployed URLs

---

## Production URLs

**Backend (Render)**: `https://moviematch-backend.onrender.com`
- API Docs: `https://moviematch-backend.onrender.com/docs`
- Health check: `https://moviematch-backend.onrender.com/health`

**Frontend (Vercel)**: `https://your-vercel-app.vercel.app`

---

## Cost

- **Render**: Free tier (512MB RAM, up to $12/month for better performance)
- **Vercel**: Free tier (unlimited deployments)
- **Total**: Free to ~$12/month

---

## Next Steps

1. Monitor performance on deployed services
2. Set up monitoring/alerts
3. Consider upgrading Render tier if needed
4. Add custom domain if desired
