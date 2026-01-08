#!/bin/bash
set -e

# ============================================
# Job Hunt Server Setup Script
# Run this once on a fresh DigitalOcean droplet
# ============================================

DOMAIN="careers.nicksolly.co.uk"
EMAIL="nick@nicksolly.co.uk"  # Change this to your email
APP_DIR="/opt/jobhunt"

echo "=== Job Hunt Server Setup ==="
echo "Domain: $DOMAIN"
echo "App Directory: $APP_DIR"
echo ""

# Update system
echo ">>> Updating system packages..."
apt-get update
apt-get upgrade -y

# Install Docker
echo ">>> Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

# Install Docker Compose plugin
echo ">>> Installing Docker Compose..."
apt-get install -y docker-compose-plugin

# Create app directory
echo ">>> Creating app directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Create required directories
mkdir -p certbot/conf certbot/www
mkdir -p nginx
mkdir -p grafana/provisioning/datasources grafana/provisioning/dashboards grafana/dashboards

# Create initial nginx config (HTTP only for certbot)
echo ">>> Creating initial nginx config..."
cat > nginx/nginx.initial.conf << 'NGINXEOF'
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name careers.nicksolly.co.uk;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 200 'Server is running. Waiting for SSL setup.';
            add_header Content-Type text/plain;
        }
    }
}
NGINXEOF

# Start nginx for certbot challenge
echo ">>> Starting nginx for SSL certificate..."
docker run -d --name nginx-init \
    -p 80:80 \
    -v $APP_DIR/nginx/nginx.initial.conf:/etc/nginx/nginx.conf:ro \
    -v $APP_DIR/certbot/www:/var/www/certbot:ro \
    nginx:alpine

# Wait for nginx to start
sleep 5

# Get SSL certificate
echo ">>> Obtaining SSL certificate..."
docker run --rm \
    -v $APP_DIR/certbot/conf:/etc/letsencrypt \
    -v $APP_DIR/certbot/www:/var/www/certbot \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN

# Stop initial nginx
echo ">>> Stopping initial nginx..."
docker stop nginx-init
docker rm nginx-init

# Setup firewall
echo ">>> Configuring firewall..."
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Create systemd service for auto-start
echo ">>> Creating systemd service..."
cat > /etc/systemd/system/jobhunt.service << 'SERVICEEOF'
[Unit]
Description=Job Hunt Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/jobhunt
ExecStart=/usr/bin/docker compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.prod.yml down

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable jobhunt

# Create SSL renewal cron job
echo ">>> Setting up SSL renewal..."
(crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/docker run --rm -v /opt/jobhunt/certbot/conf:/etc/letsencrypt -v /opt/jobhunt/certbot/www:/var/www/certbot certbot/certbot renew --quiet && docker exec jobhunt-nginx nginx -s reload") | crontab -

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Add GitHub Secrets to your repository:"
echo "   - DROPLET_SSH_KEY: Your SSH private key"
echo "   - DROPLET_IP: 129.212.162.202"
echo "   - OPENAI_API_KEY: Your OpenAI key"
echo "   - ADZUNA_APP_ID: Your Adzuna app ID"
echo "   - ADZUNA_API_KEY: Your Adzuna API key"
echo "   - APP_PASSWORD: Your login password"
echo "   - SECRET_KEY: A random 32+ character string"
echo "   - COHERE_API_KEY: (Optional) Cohere API key"
echo ""
echo "2. Push to main branch to trigger deployment"
echo ""
echo "3. Access your app at https://$DOMAIN"
