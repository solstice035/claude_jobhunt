# Deployment Guide

Complete guide to deploying the Job Hunt application to DigitalOcean.

## Prerequisites

- DigitalOcean account with a Droplet created
- Domain name with DNS access (Namecheap)
- GitHub repository with Actions enabled

## Architecture

```
Internet → Nginx (SSL) → Frontend (Next.js) / Backend (FastAPI)
                      → Redis (Cache)
                      → Grafana (Monitoring)
```

## Step 1: Create DigitalOcean Droplet

1. Log in to [DigitalOcean](https://cloud.digitalocean.com)
2. Create a new Droplet:
   - **Image:** Ubuntu 24.04 LTS
   - **Size:** Basic $12/mo (1 vCPU, 2GB RAM, 50GB SSD)
   - **Region:** London (LON1)
   - **Authentication:** SSH Key
   - **Hostname:** `jobhunt`

3. Note your Droplet IP address: `188.166.151.61`

## Step 2: Configure DNS (Namecheap)

1. Log in to [Namecheap](https://www.namecheap.com)
2. Go to Domain List → Manage → Advanced DNS
3. Add a new A Record:
   - **Host:** `careers`
   - **Value:** `188.166.151.61`
   - **TTL:** 1 min (increase to 30 min after verified)

4. Wait 5-10 minutes for DNS propagation
5. Verify: `ping careers.nicksolly.co.uk`

## Step 3: Setup Server (One-time)

SSH into your droplet and run the setup script:

```bash
ssh root@188.166.151.61

# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/solstice035/claude_jobhunt/main/scripts/setup-server.sh -o setup.sh
chmod +x setup.sh

# Edit email address in script first!
nano setup.sh  # Change EMAIL="nick@nicksolly.co.uk" to your email

# Run setup
./setup.sh
```

This script will:
- Install Docker and Docker Compose
- Obtain SSL certificate from Let's Encrypt
- Configure firewall (UFW)
- Create systemd service for auto-start
- Setup SSL auto-renewal cron job

## Step 4: Configure GitHub Secrets

Go to your repository: **Settings → Secrets and variables → Actions**

Add these secrets:

| Secret | Description | Example |
|--------|-------------|---------|
| `DROPLET_SSH_KEY` | Private SSH key for droplet | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `DROPLET_IP` | Droplet IP address | `188.166.151.61` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ADZUNA_APP_ID` | Adzuna application ID | `abc123` |
| `ADZUNA_API_KEY` | Adzuna API key | `xyz789` |
| `APP_PASSWORD` | Login password | `your-secure-password` |
| `SECRET_KEY` | JWT encryption key (32+ chars) | `your-32-char-random-string-here!` |
| `COHERE_API_KEY` | (Optional) Cohere API key | `...` |

### Generate SSH Key for GitHub Actions

```bash
# On your local machine
ssh-keygen -t ed25519 -C "github-actions" -f ~/.ssh/github_actions_deploy

# Copy public key to droplet
ssh-copy-id -i ~/.ssh/github_actions_deploy.pub root@188.166.151.61

# Copy private key content - this goes in DROPLET_SSH_KEY secret
cat ~/.ssh/github_actions_deploy
```

## Step 5: Deploy

Push to the `main` branch to trigger deployment:

```bash
git push origin main
```

Or manually trigger in GitHub: **Actions → Deploy to Production → Run workflow**

## Step 6: Verify Deployment

1. Check GitHub Actions: `https://github.com/solstice035/claude_jobhunt/actions`
2. Visit your app: `https://careers.nicksolly.co.uk`
3. Check monitoring: `https://careers.nicksolly.co.uk/grafana` (admin/admin)

## Troubleshooting

### Check container logs

```bash
ssh root@188.166.151.61
cd /opt/jobhunt
docker compose -f docker-compose.prod.yml logs -f
```

### Check specific service

```bash
docker compose -f docker-compose.prod.yml logs backend
docker compose -f docker-compose.prod.yml logs frontend
docker compose -f docker-compose.prod.yml logs nginx
```

### Restart services

```bash
docker compose -f docker-compose.prod.yml restart
```

### SSL certificate issues

```bash
# Check certificate status
docker run --rm -v /opt/jobhunt/certbot/conf:/etc/letsencrypt certbot/certbot certificates

# Force renewal
docker run --rm -v /opt/jobhunt/certbot/conf:/etc/letsencrypt -v /opt/jobhunt/certbot/www:/var/www/certbot certbot/certbot renew --force-renewal
```

### Database backup

```bash
# Backup
docker cp jobhunt-backend:/app/data/jobs.db ./jobs-backup-$(date +%Y%m%d).db

# Restore
docker cp ./jobs-backup.db jobhunt-backend:/app/data/jobs.db
docker compose -f docker-compose.prod.yml restart backend
```

## Monitoring

Access Grafana at `https://careers.nicksolly.co.uk/grafana`

Default credentials: `admin` / `admin`

Available dashboards:
- **System Overview** - Overall health
- **API Performance** - Request rates and latencies
- **Job Matching** - ML endpoint metrics
- **Cache Performance** - Redis hit rates
- **Background Tasks** - Celery task stats

## Updating

The application auto-deploys on push to `main`. To manually update:

```bash
ssh root@188.166.151.61
cd /opt/jobhunt
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
```

## Rollback

To rollback to a previous version:

```bash
# Find previous image SHA from GitHub Actions logs
# Then on server:
cd /opt/jobhunt
docker compose -f docker-compose.prod.yml down
# Edit docker-compose.prod.yml to use specific SHA tag instead of :latest
docker compose -f docker-compose.prod.yml up -d
```
