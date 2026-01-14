# SSH Guide for DigitalOcean

A practical guide to SSH (Secure Shell) for managing your DigitalOcean droplet.

## What is SSH?

SSH is a secure way to connect to a remote server's command line. Think of it as a secure tunnel that lets you type commands on your server as if you were sitting in front of it.

## Quick Reference

```bash
# Connect to your droplet
ssh root@129.212.162.202

# Connect with a specific key
ssh -i ~/.ssh/my_key root@129.212.162.202

# Copy a file TO the server
scp local_file.txt root@129.212.162.202:/path/on/server/

# Copy a file FROM the server
scp root@129.212.162.202:/path/on/server/file.txt ./local_destination/

# Copy a folder (recursive)
scp -r ./local_folder root@129.212.162.202:/path/on/server/
```

---

## Initial Setup

### 1. Generate an SSH Key (if you don't have one)

```bash
# Generate a new key pair
ssh-keygen -t ed25519 -C "your_email@example.com"

# When prompted:
# - Press Enter to accept default location (~/.ssh/id_ed25519)
# - Enter a passphrase (recommended) or press Enter for none
```

This creates two files:
- `~/.ssh/id_ed25519` - Your **private key** (NEVER share this)
- `~/.ssh/id_ed25519.pub` - Your **public key** (safe to share)

### 2. Add Your Key to DigitalOcean

**Option A: Via DigitalOcean Console (for new droplets)**

1. Go to [DigitalOcean Security Settings](https://cloud.digitalocean.com/account/security)
2. Click "Add SSH Key"
3. Paste your public key: `cat ~/.ssh/id_ed25519.pub`
4. Give it a name and save

**Option B: Copy to existing droplet**

```bash
# Automatic method (if you can already access the server)
ssh-copy-id root@129.212.162.202

# Manual method (if using console access)
# 1. Copy your public key
cat ~/.ssh/id_ed25519.pub

# 2. On the server, add it to authorized_keys
echo "your-public-key-here" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 3. Test Your Connection

```bash
ssh root@129.212.162.202
```

If successful, you'll see the server's command prompt.

---

## SSH Config File (Recommended)

Create `~/.ssh/config` to simplify connections:

```bash
# Edit or create the config file
nano ~/.ssh/config
```

Add this content:

```
Host jobhunt
    HostName 129.212.162.202
    User root
    IdentityFile ~/.ssh/id_ed25519

# You can add more servers
Host another-server
    HostName 192.168.1.100
    User admin
    IdentityFile ~/.ssh/other_key
```

Now you can simply type:

```bash
ssh jobhunt
```

Instead of:

```bash
ssh -i ~/.ssh/id_ed25519 root@129.212.162.202
```

---

## Common Tasks

### Connecting

```bash
# Basic connection
ssh root@129.212.162.202

# Using config alias
ssh jobhunt

# Run a single command without staying connected
ssh jobhunt "docker compose ps"

# Run multiple commands
ssh jobhunt "cd /opt/jobhunt && docker compose -f docker-compose.prod.yml ps"
```

### File Transfer with SCP

```bash
# Copy file TO server
scp ./local_file.txt jobhunt:/opt/jobhunt/

# Copy file FROM server
scp jobhunt:/opt/jobhunt/logs/app.log ./

# Copy entire directory TO server
scp -r ./my_folder jobhunt:/opt/jobhunt/

# Copy entire directory FROM server
scp -r jobhunt:/opt/jobhunt/data ./local_backup/
```

### File Transfer with rsync (better for large transfers)

```bash
# Sync a directory (only copies changed files)
rsync -avz ./local_folder/ jobhunt:/opt/jobhunt/folder/

# Sync with progress indicator
rsync -avz --progress ./local_folder/ jobhunt:/opt/jobhunt/folder/

# Dry run (show what would be copied)
rsync -avzn ./local_folder/ jobhunt:/opt/jobhunt/folder/
```

---

## DigitalOcean-Specific Tasks

### Access via DigitalOcean Console (Fallback)

If SSH isn't working:

1. Log into [DigitalOcean](https://cloud.digitalocean.com)
2. Go to Droplets → Select your droplet
3. Click "Access" tab
4. Click "Launch Droplet Console"

This gives you terminal access through the browser.

### Common Server Management Commands

```bash
# Check system resources
htop                    # Interactive process viewer (Ctrl+C to exit)
df -h                   # Disk space usage
free -m                 # Memory usage
uptime                  # System uptime and load

# View system logs
journalctl -f           # Follow system logs (Ctrl+C to exit)
journalctl -u nginx     # Nginx service logs

# Service management
systemctl status nginx  # Check service status
systemctl restart nginx # Restart a service
systemctl stop nginx    # Stop a service
systemctl start nginx   # Start a service

# Firewall (UFW)
ufw status              # Check firewall rules
ufw allow 22            # Allow SSH
ufw allow 80            # Allow HTTP
ufw allow 443           # Allow HTTPS
```

### Docker Commands (for this project)

```bash
# Navigate to project
cd /opt/jobhunt

# View container status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs --tail=50 backend
docker compose -f docker-compose.prod.yml logs --tail=50 frontend
docker compose -f docker-compose.prod.yml logs --tail=50 nginx

# Restart all containers
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d

# Restart a specific container
docker compose -f docker-compose.prod.yml restart backend

# Pull latest images and restart
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d

# Check Docker disk usage
docker system df

# Clean up unused Docker resources
docker system prune -a  # WARNING: Removes all unused images
```

---

## Troubleshooting

### "Permission denied (publickey)"

Your SSH key isn't authorized on the server.

```bash
# Check your key exists
ls -la ~/.ssh/

# Verify you're using the right key
ssh -v root@129.212.162.202  # Verbose mode shows which keys are tried

# Add your key via DigitalOcean Console (see fallback access above)
```

### "Connection refused"

The server isn't accepting SSH connections.

- Check if the server is running (DigitalOcean dashboard)
- Verify SSH is enabled: `systemctl status sshd` (via console)
- Check firewall: `ufw status` (via console)

### "Connection timed out"

Network issue or wrong IP address.

```bash
# Verify the IP address
ping 129.212.162.202

# Check DigitalOcean dashboard for correct IP
```

### "Host key verification failed"

The server's identity has changed (common after rebuilding a droplet).

```bash
# Remove the old key
ssh-keygen -R 129.212.162.202

# Try connecting again
ssh root@129.212.162.202
```

---

## Security Best Practices

1. **Use SSH keys, not passwords** - Keys are more secure
2. **Protect your private key** - Never share `~/.ssh/id_ed25519`
3. **Use a passphrase** - Adds extra protection if your key is stolen
4. **Disable root login** (optional) - Create a regular user and use `sudo`
5. **Keep software updated** - Run `apt update && apt upgrade` regularly

### Setting Up a Non-Root User (Recommended)

```bash
# Create a new user
adduser deploy

# Add to sudo group
usermod -aG sudo deploy

# Copy SSH key to new user
mkdir -p /home/deploy/.ssh
cp ~/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys

# Now you can login as: ssh deploy@129.212.162.202
```

---

## Quick Cheat Sheet

| Task | Command |
|------|---------|
| Connect | `ssh jobhunt` |
| Copy file to server | `scp file.txt jobhunt:/path/` |
| Copy file from server | `scp jobhunt:/path/file.txt ./` |
| Run remote command | `ssh jobhunt "command"` |
| View Docker status | `ssh jobhunt "cd /opt/jobhunt && docker compose -f docker-compose.prod.yml ps"` |
| Restart Docker stack | `ssh jobhunt "cd /opt/jobhunt && docker compose -f docker-compose.prod.yml down && docker compose -f docker-compose.prod.yml up -d"` |
| View logs | `ssh jobhunt "cd /opt/jobhunt && docker compose -f docker-compose.prod.yml logs --tail=100"` |
| Check disk space | `ssh jobhunt "df -h"` |
| Check memory | `ssh jobhunt "free -m"` |

---

## Your Server Details

| Item | Value |
|------|-------|
| IP Address | `129.212.162.202` |
| Username | `root` |
| App Location | `/opt/jobhunt` |
| Compose File | `docker-compose.prod.yml` |
| Domain | `careers.nicksolly.co.uk` |

---

## Related Documentation

- [OPERATIONS.md](OPERATIONS.md) - Troubleshooting guide and operational procedures
