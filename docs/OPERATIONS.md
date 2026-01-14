# Operations Runbook

A practical guide for managing and troubleshooting the Job Search Agent in production.

## Quick Reference

```bash
# SSH into server (after setting up ~/.ssh/config)
ssh jobhunt

# Or directly
ssh root@129.212.162.202

# Navigate to app
cd /opt/jobhunt

# All docker commands use this prefix
docker compose -f docker-compose.prod.yml <command>
```

---

## Common Operations

### Check Status

```bash
# View all containers
docker compose -f docker-compose.prod.yml ps

# Expected output - all should be "Up" and healthy:
# jobhunt-backend    Up (healthy)
# jobhunt-frontend   Up (healthy)
# jobhunt-nginx      Up
# jobhunt-redis      Up (healthy)
# jobhunt-prometheus Up
# jobhunt-grafana    Up
```

### View Logs

```bash
# All services
docker compose -f docker-compose.prod.yml logs --tail=50

# Specific service
docker compose -f docker-compose.prod.yml logs --tail=50 backend
docker compose -f docker-compose.prod.yml logs --tail=50 frontend
docker compose -f docker-compose.prod.yml logs --tail=50 nginx

# Follow logs in real-time (Ctrl+C to exit)
docker compose -f docker-compose.prod.yml logs -f backend
```

### Restart Services

```bash
# IMPORTANT: Use down/up, NOT restart, when:
# - You've changed .env files
# - You're seeing 502 errors
# - Containers show "unhealthy"

# Full restart (recommended)
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d

# Quick restart (only if no config changes)
docker compose -f docker-compose.prod.yml restart backend
```

### Update to Latest Version

```bash
# Pull new images and restart
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d
```

---

## Troubleshooting Guide

### 502 Bad Gateway

**Symptoms:**
- Website shows "502 Bad Gateway" nginx error
- Both frontend and API return 502

**Diagnosis:**
```bash
# Check container status
docker compose -f docker-compose.prod.yml ps

# Look for "unhealthy" status or containers that are down
# Check nginx logs for connection errors
docker compose -f docker-compose.prod.yml logs --tail=50 nginx
```

**Common causes:**

1. **Stale Docker DNS** (most common)
   - Nginx caches container IPs
   - If containers restart, IPs may change but nginx has old addresses

   **Fix:**
   ```bash
   docker compose -f docker-compose.prod.yml down
   docker compose -f docker-compose.prod.yml up -d
   ```

2. **Container crashed**
   - Check if backend/frontend are running

   **Fix:**
   ```bash
   # Check logs for crash reason
   docker compose -f docker-compose.prod.yml logs --tail=100 backend
   docker compose -f docker-compose.prod.yml logs --tail=100 frontend

   # Restart
   docker compose -f docker-compose.prod.yml down
   docker compose -f docker-compose.prod.yml up -d
   ```

3. **Out of memory**
   ```bash
   free -m
   docker stats --no-stream
   ```

   **Fix:** May need to increase droplet size or reduce container memory limits.

---

### Login/Password Not Working

**Symptoms:**
- Login returns "Invalid password" or 401 Unauthorized
- Password was recently changed

**Diagnosis:**
```bash
# Check what's in .env file
cat /opt/jobhunt/.env | grep -i password

# Check what the container actually sees
docker compose -f docker-compose.prod.yml exec backend env | grep -i password
```

**Common causes:**

1. **`.env` changes not loaded**

   `docker compose restart` does NOT reload .env files!

   **Fix:**
   ```bash
   docker compose -f docker-compose.prod.yml down
   docker compose -f docker-compose.prod.yml up -d
   ```

2. **Special characters in password**

   Dollar signs (`$`) are interpreted as variable references in .env files.

   **Fix options:**
   - Escape with double dollar: `APP_PASSWORD=$$MyPassword`
   - Or use a password without `$` at the start

3. **Whitespace in .env file**

   Extra spaces around `=` will break things.

   **Wrong:** `APP_PASSWORD = mypassword`
   **Right:** `APP_PASSWORD=mypassword`

---

### Container Shows "Unhealthy"

**Symptoms:**
- `docker compose ps` shows "(unhealthy)" status
- Service may still be running but failing health checks

**Diagnosis:**
```bash
# Check health check configuration and recent results
docker inspect jobhunt-backend | grep -A 20 "Health"

# Check container logs
docker compose -f docker-compose.prod.yml logs --tail=100 backend
```

**Common causes:**

1. **Health check endpoint not responding**
   - Application may be overloaded or stuck

   **Fix:**
   ```bash
   docker compose -f docker-compose.prod.yml restart backend
   ```

2. **Health check timing**
   - Container may still be starting up
   - Wait 30-60 seconds and check again

---

### Database Issues

**Symptoms:**
- Jobs not loading
- Errors mentioning SQLite or database

**Diagnosis:**
```bash
# Check if database volume exists
docker volume ls | grep backend-data

# Check database file inside container
docker compose -f docker-compose.prod.yml exec backend ls -la /app/data/
```

**Fix:**
```bash
# If database is corrupted, you may need to reset it
# WARNING: This deletes all job data

docker compose -f docker-compose.prod.yml down
docker volume rm jobhunt_backend-data
docker compose -f docker-compose.prod.yml up -d
```

---

### SSL Certificate Issues

**Symptoms:**
- Browser shows "Not Secure" warning
- Certificate expired errors

**Diagnosis:**
```bash
# Check certificate expiry
docker compose -f docker-compose.prod.yml exec certbot certbot certificates

# Check certbot logs
docker compose -f docker-compose.prod.yml logs certbot
```

**Fix:**
```bash
# Force certificate renewal
docker compose -f docker-compose.prod.yml exec certbot certbot renew --force-renewal

# Restart nginx to pick up new cert
docker compose -f docker-compose.prod.yml restart nginx
```

---

### Disk Space Issues

**Symptoms:**
- Containers failing to start
- Write errors in logs

**Diagnosis:**
```bash
# Check system disk space
df -h

# Check Docker disk usage
docker system df

# Find large files
du -sh /var/lib/docker/*
```

**Fix:**
```bash
# Remove unused Docker resources (safe)
docker system prune

# Remove unused images too (more aggressive)
docker system prune -a

# Remove old logs
truncate -s 0 /var/lib/docker/containers/*/*-json.log
```

---

## Environment Variables

The `.env` file at `/opt/jobhunt/.env` contains:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | For job matching embeddings |
| `ADZUNA_APP_ID` | Adzuna API credentials |
| `ADZUNA_API_KEY` | Adzuna API credentials |
| `APP_PASSWORD` | Login password |
| `SECRET_KEY` | JWT session encryption |
| `GRAFANA_PASSWORD` | Grafana admin password (optional) |

**Remember:** After changing `.env`, always do a full `down && up`:

```bash
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d
```

---

## Monitoring

### Grafana Dashboard

Access at: `https://careers.nicksolly.co.uk/grafana`
- Default user: `admin`
- Password: Set in `.env` as `GRAFANA_PASSWORD` (default: `admin`)

### Prometheus Metrics

Backend exposes metrics at `/metrics` endpoint.

### Manual Health Checks

```bash
# Check backend API
curl -s https://careers.nicksolly.co.uk/api/docs | head -20

# Check frontend
curl -s https://careers.nicksolly.co.uk | head -20

# From inside server (bypassing nginx)
docker compose -f docker-compose.prod.yml exec nginx wget -qO- http://backend:8000/docs | head -20
docker compose -f docker-compose.prod.yml exec nginx wget -qO- http://frontend:3000 | head -20
```

---

## Deployment

Deployments happen automatically via GitHub Actions when you push to `main`.

### Manual Deployment

If you need to deploy manually:

```bash
cd /opt/jobhunt

# Pull latest images
docker compose -f docker-compose.prod.yml pull

# Restart with new images
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d

# Verify
docker compose -f docker-compose.prod.yml ps
```

### Rollback

If a new deployment breaks things:

```bash
# Find previous image tags
docker images | grep jobhunt

# Edit docker-compose.prod.yml to use specific tag instead of :latest
# Then restart
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d
```

---

## Server Maintenance

### System Updates

```bash
# Check for updates
apt update
apt list --upgradable

# Install updates
apt upgrade -y

# If kernel updated, reboot required
reboot
```

### Backups

The database is stored in a Docker volume. To backup:

```bash
# Create backup directory
mkdir -p /opt/backups

# Backup database
docker compose -f docker-compose.prod.yml exec backend cat /app/data/jobs.db > /opt/backups/jobs-$(date +%Y%m%d).db

# Or copy the entire volume
docker run --rm -v jobhunt_backend-data:/data -v /opt/backups:/backup alpine tar czf /backup/backend-data-$(date +%Y%m%d).tar.gz /data
```

---

## Quick Diagnostic Script

Run this to get a full status overview:

```bash
echo "=== Container Status ==="
docker compose -f docker-compose.prod.yml ps

echo -e "\n=== Disk Space ==="
df -h /

echo -e "\n=== Memory ==="
free -m

echo -e "\n=== Docker Disk ==="
docker system df

echo -e "\n=== Recent Backend Errors ==="
docker compose -f docker-compose.prod.yml logs --tail=20 backend | grep -i error

echo -e "\n=== Recent Nginx Errors ==="
docker compose -f docker-compose.prod.yml logs --tail=20 nginx | grep -i error
```

---

## Contact & Resources

| Resource | Location |
|----------|----------|
| Source Code | https://github.com/solstice035/claude_jobhunt |
| Domain | careers.nicksolly.co.uk |
| Server IP | 129.212.162.202 |
| DigitalOcean Console | https://cloud.digitalocean.com |
| App Directory | /opt/jobhunt |
| Compose File | docker-compose.prod.yml |
