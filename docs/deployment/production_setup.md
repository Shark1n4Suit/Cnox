# Production Deployment Guide

This guide covers deploying the SAIC runner as a production service on Linux servers using systemd for reliable, secure operation.

## Prerequisites

- Linux server (Ubuntu 20.04+, CentOS 8+, or equivalent)
- Root or sudo access
- Network connectivity for model downloads
- Sufficient storage for AI models
- GPU support (optional, for TensorRT backend)

## Installation

### 1. Download and Install SAIC Runner

```bash
# Download the latest SAIC runner
curl -fsSL https://install.coelanox.com/saic | sh

# Verify installation
saic-runner --version
```

### 2. Create System User

```bash
# Create dedicated system user for security
sudo useradd -r -s /bin/false saic-runner

# Create application directory
sudo mkdir -p /opt/saic
sudo chown saic-runner:saic-runner /opt/saic
```

### 3. Create Configuration Directory

```bash
# Create configuration directory
sudo mkdir -p /etc/saic
sudo chown saic-runner:saic-runner /etc/saic

# Set secure permissions
sudo chmod 750 /etc/saic
```

## Systemd Service Setup

### 1. Create Service File

Create `/etc/systemd/system/saic-runner.service`:

```ini
[Unit]
Description=SAIC Runner Service
Documentation=https://github.com/coelanox/saic
After=network.target

[Service]
Type=simple
User=saic-runner
Group=saic-runner
WorkingDirectory=/opt/saic
ExecStart=/usr/local/bin/saic-runner daemon --config /etc/saic/config.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=saic-runner

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/saic /var/log/saic

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

### 2. Create Configuration File

Create `/etc/saic/config.yaml`:

```yaml
# SAIC Runner Production Configuration
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  
security:
  # Enable DNA key validation
  validate_signatures: true
  
  # Cryptographic settings
  encryption_algorithm: "aes-256-gcm"
  
  # Access control
  allowed_origins:
    - "https://your-domain.com"
    - "https://api.your-domain.com"
  
logging:
  level: "info"
  format: "json"
  output: "/var/log/saic/runner.log"
  
backends:
  cpu:
    enabled: true
    backend: "openvino"
    threads: 4
    
  gpu:
    enabled: true
    backend: "tensorrt"
    device_id: 0
    
storage:
  model_cache: "/opt/saic/models"
  temp_directory: "/opt/saic/temp"
  
monitoring:
  metrics_enabled: true
  health_check_interval: 30
  
# Production security settings
production:
  # Disable debug endpoints
  debug_enabled: false
  
  # Rate limiting
  rate_limit:
    requests_per_minute: 1000
    burst_size: 100
    
  # Request size limits
  max_request_size: "100MB"
  max_model_size: "10GB"
```

### 3. Create Log Directory

```bash
# Create log directory
sudo mkdir -p /var/log/saic
sudo chown saic-runner:saic-runner /var/log/saic
sudo chmod 750 /var/log/saic
```

### 4. Enable and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable saic-runner

# Start the service
sudo systemctl start saic-runner

# Check service status
sudo systemctl status saic-runner
```

## Security Configuration

### 1. Firewall Setup

```bash
# Allow SAIC runner port (adjust as needed)
sudo ufw allow 8080/tcp

# For production, consider using a reverse proxy
sudo ufw allow 443/tcp
```

### 2. Reverse Proxy Configuration (Recommended)

Create `/etc/nginx/sites-available/saic-runner`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 3. SSL Certificate Setup

```bash
# Install Certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com
```

## Monitoring and Maintenance

### 1. Service Management

```bash
# Check service status
sudo systemctl status saic-runner

# View logs
sudo journalctl -u saic-runner -f

# Restart service
sudo systemctl restart saic-runner

# Reload configuration
sudo systemctl reload saic-runner
```

### 2. Health Check Endpoint

The SAIC runner provides a health check endpoint:

```bash
# Check service health
curl http://localhost:8080/health

# Expected response
{
  "status": "healthy",
  "version": "v0.5.0-beta",
  "backends": {
    "cpu": "available",
    "gpu": "available"
  }
}
```

### 3. Log Monitoring

```bash
# Monitor application logs
sudo tail -f /var/log/saic/runner.log

# Monitor system logs
sudo journalctl -u saic-runner --since "1 hour ago"
```

### 4. Performance Monitoring

```bash
# Check resource usage
sudo systemctl status saic-runner --no-pager -l

# Monitor system resources
htop
iotop
```

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check configuration syntax
   saic-runner validate-config /etc/saic/config.yaml
   
   # Check logs for errors
   sudo journalctl -u saic-runner --no-pager
   ```

2. **Permission denied errors**
   ```bash
   # Fix ownership
   sudo chown -R saic-runner:saic-runner /opt/saic
   sudo chown -R saic-runner:saic-runner /var/log/saic
   ```

3. **GPU backend not available**
   ```bash
   # Check GPU drivers
   nvidia-smi
   
   # Verify TensorRT installation
   saic-runner list-backends
   ```

### Performance Tuning

1. **Adjust worker processes**
   ```yaml
   server:
     workers: 8  # Increase for high-traffic deployments
   ```

2. **Optimize memory usage**
   ```yaml
   backends:
     cpu:
       threads: 8  # Match CPU cores
   ```

3. **Enable model caching**
   ```yaml
   storage:
     model_cache: "/opt/saic/models"
     cache_size: "10GB"
   ```

## Backup and Recovery

### 1. Configuration Backup

```bash
# Backup configuration
sudo tar -czf saic-config-backup-$(date +%Y%m%d).tar.gz /etc/saic
```

### 2. Model Cache Backup

```bash
# Backup model cache
sudo tar -czf saic-models-backup-$(date +%Y%m%d).tar.gz /opt/saic/models
```

### 3. Recovery Procedure

```bash
# Restore configuration
sudo tar -xzf saic-config-backup-YYYYMMDD.tar.gz -C /

# Restore models
sudo tar -xzf saic-models-backup-YYYYMMDD.tar.gz -C /

# Restart service
sudo systemctl restart saic-runner
```

## Security Best Practices

1. **Regular Updates**: Keep SAIC runner updated to latest version
2. **Access Control**: Use firewall rules and reverse proxy
3. **SSL/TLS**: Always use HTTPS in production
4. **Monitoring**: Set up log monitoring and alerting
5. **Backups**: Regular backups of configuration and models
6. **Audit**: Regular security audits and penetration testing

---

**Next Steps**: After deployment, test the service thoroughly and monitor performance. Consider setting up monitoring tools like Prometheus and Grafana for production environments. 