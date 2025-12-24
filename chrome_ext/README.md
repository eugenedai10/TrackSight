# Closest Links Highlighter - Chrome Extension

A Chrome extension that highlights the 3 closest links to your mouse pointer with distinct colors. Includes native messaging support to control link clicks from external Python applications.

## Features

- **Real-time Highlighting**: Highlights the 3 closest links to your mouse cursor as you move it around the page
- **Color Persistence**: Each link maintains its assigned color (red, green, or blue) while it remains in the top 3
- **Smart Color Reuse**: When a link exits the top 3, its color is reassigned to a new link entering the top 3
- **Throttled Performance**: Uses a 50ms throttle to ensure smooth performance even on pages with many links
- **Viewport Aware**: Only highlights links currently visible in the viewport
- **Auto-Activated**: Extension is active by default on all pages
- **Persistent State**: Activation state is saved globally - toggle once and it applies everywhere
- **Keyboard Toggle**: Toggle on/off with `Ctrl+Shift+L` (Windows/Linux) or `Cmd+Shift+L` (Mac)
- **Synced Across Tabs**: State changes are instantly synced across all open tabs

## Installation

1. **Download the Extension**
   - Clone or download this repository to your local machine

2. **Open Chrome Extensions Page**
   - Open Google Chrome
   - Navigate to `chrome://extensions/`
   - Or click the three-dot menu → Extensions → Manage Extensions

3. **Enable Developer Mode**
   - Toggle the "Developer mode" switch in the top-right corner

4. **Load the Extension**
   - Click "Load unpacked"
   - Select the `mouse_links` folder containing the extension files
   - The extension should now appear in your extensions list

## Usage

1. **Navigate to any webpage** with links
2. **Move your mouse** - the extension is **active by default** and will automatically highlight the 3 closest links with colored borders:
   - 🔴 Red border
   - 🟢 Green border
   - 🔵 Blue border
3. **Press `Ctrl+Shift+L`** (or `Cmd+Shift+L` on Mac) to toggle highlighting on/off
   - Your preference is saved globally across all websites
   - Changes sync instantly to all open tabs

## How It Works

### Distance Calculation
The extension calculates the Euclidean distance from your mouse pointer to the closest edge of each link's bounding box. This ensures accurate distance measurement regardless of link size or position.

### Color Assignment Logic
- 3 colors are available: Red, Green, and Blue
- When a link enters the top 3, it's assigned an available color
- If a link was previously in the top 3 and still is, it keeps its color
- When a link exits the top 3, its color becomes available for the next link

### Performance Optimization
- Mouse movements are throttled to 50ms intervals
- Only visible links in the viewport are highlighted
- Efficient DOM queries and distance calculations

## Technical Details

- **Manifest Version**: 3
- **Permissions**: activeTab (minimal permissions required)
- **Technologies**: Vanilla JavaScript, Chrome Extension APIs
- **Throttle Interval**: 50ms
- **Supported Browsers**: Google Chrome (and Chromium-based browsers)

## Files

- `manifest.json` - Extension configuration and metadata
- `background.js` - Service worker handling keyboard shortcuts
- `content.js` - Main logic for link detection and highlighting
- `README.md` - This file

## Troubleshooting

**Extension not working?**
- Make sure you've pressed the keyboard shortcut to activate it
- Check the browser console (F12) for any error messages
- Ensure the extension is enabled in `chrome://extensions/`

**Links not highlighting?**
- Verify the page has `<a>` tags with `href` attributes
- Check if the links are visible in the viewport
- Try refreshing the page

**Keyboard shortcut not working?**
- Check if another extension is using the same shortcut
- Go to `chrome://extensions/shortcuts` to view/modify shortcuts

## Native Messaging (Python Integration)

This extension supports native messaging, allowing Python applications to programmatically click the highlighted links.

### Quick Setup

1. **Install the native messaging host:**
   ```bash
   bash install_native_host.sh
   ```
   Follow the prompts and enter your Chrome Extension ID when asked.

2. **Test it:**
   ```bash
   # Open test.html, activate extension (Cmd+Shift+L), then:
   python3 send_link_command.py 1  # Click red link
   python3 send_link_command.py 2  # Click green link
   python3 send_link_command.py 3  # Click blue link
   ```

### Documentation

See **[NATIVE_MESSAGING.md](NATIVE_MESSAGING.md)** for complete setup instructions, troubleshooting, and advanced usage.

## Project Files

### Core Extension
- `manifest.json` - Extension configuration
- `background.js` - Service worker (keyboard shortcuts + native messaging)
- `content.js` - Main logic (link detection, highlighting, clicking)
- `test.html` - Test page with various link layouts

### Native Messaging
- `native_host.py` - Native messaging host (Python)
- `com.closest_links.host.json` - Native host manifest template
- `install_native_host.sh` - Installation script for macOS
- `send_link_command.py` - Example Python app to control links

### Documentation
- `README.md` - This file (getting started guide)
- `NATIVE_MESSAGING.md` - Native messaging setup guide
- `DEBUGGING.md` - Troubleshooting guide

## Quick Start Summary

1. **Install extension** in Chrome (Developer mode, Load unpacked)
2. **Move mouse** - links are highlighted automatically
3. **(Optional)** Toggle with `Cmd+Shift+L` (Mac) or `Ctrl+Shift+L` (Windows/Linux)
4. **(Optional)** Set up native messaging to control from Python

## License

MIT License - Feel free to use and modify as needed.
