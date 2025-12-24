#!/bin/bash
# Installation script for Native Messaging Host on macOS
# This script sets up the native messaging host for the Closest Links Highlighter extension

set -e

echo "=========================================="
echo "Closest Links Highlighter - Native Host Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MANIFEST_FILE="$SCRIPT_DIR/com.closest_links.host.json"
NATIVE_HOST="$SCRIPT_DIR/native_host.py"
TARGET_DIR="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"

# Check if native_host.py exists
if [ ! -f "$NATIVE_HOST" ]; then
    echo "ERROR: native_host.py not found in $SCRIPT_DIR"
    exit 1
fi

# Make native_host.py executable
echo "Making native_host.py executable..."
chmod +x "$NATIVE_HOST"

# Get Extension ID
echo ""
echo "To complete the installation, you need your Chrome extension ID."
echo "Follow these steps to find it:"
echo ""
echo "1. Open Chrome and go to: chrome://extensions/"
echo "2. Make sure 'Developer mode' is enabled (toggle in top-right)"
echo "3. Find 'Closest Links Highlighter' in the list"
echo "4. Copy the Extension ID (it looks like: abcdefghijklmnopqrstuvwxyz123456)"
echo ""
read -p "Enter your Extension ID: " EXTENSION_ID

# Validate extension ID (basic check - should be 32 characters)
if [ ${#EXTENSION_ID} -ne 32 ]; then
    echo "WARNING: Extension ID should be 32 characters long. You entered: ${#EXTENSION_ID} characters"
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

# Update manifest with extension ID
echo ""
echo "Updating manifest with Extension ID..."
TEMP_MANIFEST="$SCRIPT_DIR/com.closest_links.host.json.tmp"
sed "s/EXTENSION_ID_PLACEHOLDER/$EXTENSION_ID/g" "$MANIFEST_FILE" > "$TEMP_MANIFEST"

# Update path to use absolute path
sed -i '' "s|/Users/kendai/workspaces/mouse_links/native_host.py|$NATIVE_HOST|g" "$TEMP_MANIFEST"

# Create target directory if it doesn't exist
echo "Creating Native Messaging Hosts directory..."
mkdir -p "$TARGET_DIR"

# Copy manifest to target directory
echo "Installing native messaging host manifest..."
cp "$TEMP_MANIFEST" "$TARGET_DIR/com.closest_links.host.json"
rm "$TEMP_MANIFEST"

echo ""
echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "Native messaging host installed at:"
echo "  $TARGET_DIR/com.closest_links.host.json"
echo ""
echo "Next steps:"
echo "1. Reload the Chrome extension (chrome://extensions/ → click reload icon)"
echo "2. Test with: python3 send_link_command.py 1"
echo ""
echo "Logs will be written to: ~/closest_links_host.log"
echo ""
