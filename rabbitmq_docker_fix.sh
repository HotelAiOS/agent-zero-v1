#!/bin/bash
# RabbitMQ Docker Hub Connection Fix
# Dla Agent Zero V1 - Arch Linux + Fish Shell

echo "🔧 Fixing Docker Hub connection issues..."

# 1. Check if image already exists locally
if docker images | grep -q "rabbitmq.*3.12-management-alpine"; then
    echo "✅ RabbitMQ image already exists locally"
    echo "🚀 Starting existing container..."
    docker-compose up -d rabbitmq
    exit 0
fi

# 2. Try different Docker registries
echo "📡 Trying alternative registries..."

# Try with longer timeout
export DOCKER_CLIENT_TIMEOUT=300
export COMPOSE_HTTP_TIMEOUT=300

# Method 1: Pull with retries
echo "🔄 Attempting pull with retries..."
for i in {1..3}; do
    echo "Attempt $i/3..."
    if docker pull rabbitmq:3.12-management-alpine; then
        echo "✅ Successfully pulled RabbitMQ image"
        docker-compose up -d rabbitmq
        exit 0
    fi
    echo "❌ Attempt $i failed, waiting 10 seconds..."
    sleep 10
done

# Method 2: Use different tag
echo "🔄 Trying different RabbitMQ tag..."
if docker pull rabbitmq:3-management; then
    echo "✅ Successfully pulled alternative RabbitMQ image"
    # Update docker-compose to use different tag
    sed -i 's/rabbitmq:3.12-management-alpine/rabbitmq:3-management/g' docker-compose.yml
    docker-compose up -d rabbitmq
    exit 0
fi

# Method 3: Use local RabbitMQ setup (fallback)
echo "🔄 Setting up local RabbitMQ as fallback..."
echo "Installing RabbitMQ with pacman (Arch Linux)..."

# Check if running on Arch Linux
if command -v pacman &> /dev/null; then
    sudo pacman -S --noconfirm rabbitmq
    sudo systemctl enable rabbitmq
    sudo systemctl start rabbitmq

    # Configure for Agent Zero
    sudo rabbitmqctl add_user agent_zero agent_zero_rabbit_dev
    sudo rabbitmqctl set_user_tags agent_zero administrator
    sudo rabbitmqctl add_vhost agent_zero_vhost
    sudo rabbitmqctl set_permissions -p agent_zero_vhost agent_zero ".*" ".*" ".*"

    echo "✅ Local RabbitMQ setup completed"
    echo "🌐 Management UI: http://localhost:15672"
    echo "👤 User: agent_zero"
    echo "🔑 Pass: agent_zero_rabbit_dev"
    exit 0
fi

echo "❌ All methods failed. Please check internet connection."
exit 1
