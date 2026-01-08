# AI Job Search Agent

Personal job search aggregator with AI-powered matching.

## Quick Start (Development)

1. Copy `.env.example` to `.env` and fill in your API keys
2. Run: `docker compose -f docker-compose.dev.yml up --build`
3. Frontend: http://localhost:3000
4. Backend API: http://localhost:8000/docs

## Production Deployment

1. Configure `.env` with production values
2. Set up SSL certificates in `nginx/ssl/`
3. Run: `docker compose up -d --build`

## API Keys Needed

- **OpenAI**: https://platform.openai.com/api-keys
- **Adzuna**: https://developer.adzuna.com/
