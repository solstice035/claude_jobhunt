# Deployment Pipeline Design

> **For Claude:** This design document captures the CI/CD deployment architecture for the Job Hunt application.

**Goal:** Automated deployment to DigitalOcean via GitHub Actions on every push to main.

**Architecture:** Docker-based deployment with nginx reverse proxy, SSL via Let's Encrypt, and GitHub Container Registry for images.

**Tech Stack:** GitHub Actions, Docker, nginx, certbot, DigitalOcean

---

## Infrastructure

| Component | Value |
|-----------|-------|
| Provider | DigitalOcean |
| Droplet | Basic $12/mo (1 vCPU, 2GB RAM, 50GB SSD) |
| IP | 129.212.162.202 |
| Domain | careers.nicksolly.co.uk |
| DNS | Namecheap |
| Registry | ghcr.io/solstice035/claude_jobhunt |

## Pipeline Flow

```
Push to main
    ↓
GitHub Actions: Test
    - pytest (backend)
    - npm run build (frontend)
    ↓
GitHub Actions: Build
    - Build backend image
    - Build frontend image
    - Push to ghcr.io
    ↓
GitHub Actions: Deploy
    - SSH to droplet
    - docker compose pull
    - docker compose up -d
    - Health check
```

## GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `DROPLET_SSH_KEY` | Private SSH key for droplet access |
| `DROPLET_IP` | 129.212.162.202 |
| `OPENAI_API_KEY` | OpenAI API key for embeddings |
| `ADZUNA_APP_ID` | Adzuna application ID |
| `ADZUNA_API_KEY` | Adzuna API key |
| `APP_PASSWORD` | Application login password |
| `SECRET_KEY` | JWT session encryption key |
| `COHERE_API_KEY` | (Optional) Cohere re-ranking API key |

## Files Created

1. `.github/workflows/deploy.yml` - CI/CD pipeline
2. `docker-compose.prod.yml` - Production Docker config
3. `nginx/nginx.prod.conf` - SSL + reverse proxy
4. `scripts/setup-server.sh` - One-time server setup
5. `frontend/Dockerfile` - Production frontend build
6. `docs/deployment.md` - Setup guide
