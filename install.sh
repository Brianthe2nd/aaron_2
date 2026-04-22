#!/bin/bash
set -e  # exit if any command fails

# create_swap.sh - Script to create and enable a 4GB swap file

SWAPFILE=/swapfile
SIZE=4G

# Check if swapfile already exists
if [ -f "$SWAPFILE" ]; then
    echo "Swap file $SWAPFILE already exists."
    if swapon --show | grep -q "$SWAPFILE"; then
        echo "Swap file is already active. Skipping creation."
    else
        echo "Swap file exists but is not active. Enabling..."
        sudo swapon $SWAPFILE
    fi
else
    # Create the swap file
    echo "Creating swap file of size $SIZE at $SWAPFILE..."
    sudo fallocate -l $SIZE $SWAPFILE

    # Set correct permissions
    echo "Setting permissions..."
    sudo chmod 600 $SWAPFILE

    # Format the file for swap
    echo "Formatting as swap..."
    sudo mkswap $SWAPFILE

    # Enable the swap
    echo "Enabling swap..."
    sudo swapon $SWAPFILE

    # Add entry to /etc/fstab if not already present
    if ! grep -q "$SWAPFILE" /etc/fstab; then
        echo "Adding to /etc/fstab for persistence..."
        echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab
    fi
fi

# Show swap status
echo "Swap check complete. Current swap status:"
sudo swapon --show
free -h

# Install git and Python dependencies for Linux Mint (Debian/Ubuntu based)
echo "Updating package list..."
sudo apt update -y

echo "Installing Git, Python3, and pip..."
sudo apt install -y git python3 python3-pip

echo "Reinstalling pkg resources..."
sudo apt install --reinstall -y python3-setuptools

echo "Printing git version"
git --version

# Clone the repos (replace with your repo URLs)
REPO_URL="https://github.com/Brianthe2nd/aaron_2.git"

REPO_NAME=$(basename "$REPO_URL" .git)

# Clone/update main repo
if [ -d "$REPO_NAME" ]; then
    echo "Repo already cloned. Pulling latest changes..."
    cd "$REPO_NAME"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_NAME"
fi

echo "Installing yt-dlp"
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
sudo chmod a+rx /usr/local/bin/yt-dlp

echo "Installing python3-venv for Linux Mint..."
sudo apt install -y python3-venv

echo "Installing OpenGL library for Linux Mint..."
sudo apt install libgl1-mesa-dev libglu1-mesa-dev


echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing setup tools"
pip3 install --upgrade setuptools

echo "Installing camgear (vidgear)"
pip install -U vidgear[core]

echo "Installing requirements"
pip install -r requirements.txt

# Run driver.py in background
echo "Running driver.py in background..."
nohup .venv/bin/python driver.py > main.log 2>&1 &

echo "Installation complete! Check main.log for output."