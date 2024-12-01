#!/bin/bash

# Ensure the script is being run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Please run it with elevated permissions."
    exit 1
fi

# Update package lists
echo "Updating package lists..."
apt update -y

# Install necessary packages
echo "Installing tmux, htop, and nvtop..."
apt install -y tmux htop nvtop

# Verify installations
echo "Verifying installations..."
for cmd in tmux htop nvtop; do
    if command -v $cmd >/dev/null 2>&1; then
        echo "$cmd is installed successfully!"
    else
        echo "Error: $cmd installation failed."
    fi
done

echo "Setup completed!"
