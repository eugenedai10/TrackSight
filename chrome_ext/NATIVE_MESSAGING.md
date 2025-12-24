# Native Messaging Setup Guide (macOS)

This guide explains how to set up native messaging to allow Python applications to control the Chrome extension and click links programmatically.

## Overview

The native messaging architecture allows external applications (like Python scripts) to communicate with the Chrome extension:

```
Python Application (send_link_command.py)
    ↓ (Socket: localhost:9999)
Native Messaging Host (native_host.py - started by Chrome)
    ↓ (stdin/stdout)
Chrome Extension (background.js)
    ↓ (Message passing)
Content Script (content.js)
    ↓
Clicks the colored link on the webpage
```

**Key Architecture Point:** Chrome starts and manages the native host. External Python applications connect to it via a socket server running on localhost:9999.

## Prerequisites

- macOS
- Python 3 (already installed on macOS)
- Chrome with the Closest Links Highlighter extension installed
- Extension must be loaded in Developer mode

## Installation Steps

### Step 1: Find Your Extension ID

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right corner)
3. Find **Closest Links Highlighter** in the list
4. Copy the **Extension ID** (32-character string like `abcdefghijklmnopqrstuvwxyz123456`)

### Step 2: Run the Installation Script

Open Terminal and navigate to the extension directory:

```bash
cd /Users/kendai/workspaces/mouse_links
bash install_native_host.sh
```

The script will:
1. Make `native_host.py` executable
2. Ask for your Extension ID
3. Update the manifest file with your Extension ID
4. Install the native messaging host to `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/`

### Step 3: Reload the Extension

1. Go to `chrome://extensions/`
2. Find **Closest Links Highlighter**
3. Click the **reload** icon (circular arrow)

### Step 4: Verify Installation

Check the service worker console:
1. Go to `chrome://extensions/`
2. Find your extension
3. Click **"Inspect views: service worker"**
4. You should see: `"Connected to native messaging host: com.closest_links.host"`

If you see an error about the native host, check the troubleshooting section below.

## Usage

### Basic Usage

1. **Open a webpage** with links (or use `test.html`)
2. **Activate the extension** by pressing `Cmd+Shift+L`
3. **Move your mouse** to highlight the 3 closest links
4. **Run the Python command** to click a link:

```bash
# Click the RED link (1)
python3 send_link_command.py 1

# Click the GREEN link (2)
python3 send_link_command.py 2

# Click the BLUE link (3)
python3 send_link_command.py 3
```

### Color Mapping

- **1** = Red link (first closest)
- **2** = Green link (second closest)
- **3** = Blue link (third closest)

## How It Works

### Message Flow

1. **Python app** connects to socket (localhost:9999)
2. **Python app** sends JSON: `{"command": "click-link", "number": 2}`
3. **Native host** receives via socket, forwards to Chrome via stdout
4. **Background script** receives message and forwards to active tab
5. **Content script** finds the link with matching color (green = #00FF00)
6. **Content script** simulates a click on that link
7. **Response** flows back through the chain

**Debug Logs:**
- Native host: `~/closest_links_host.log`
- Background: chrome://extensions/ → Inspect service worker
- Content: F12 Console on webpage

### Communication Protocols

**Python App ↔ Native Host (Socket):**
- TCP socket on localhost:9999
- JSON messages sent as UTF-8 strings
- Simple request/response

**Native Host ↔ Chrome (Native Messaging):**
- Messages preceded by 4-byte length (uint32, little-endian)
- Message content is JSON encoded as UTF-8
- Communication over stdin/stdout
- Chrome starts and manages the native host process

## File Structure

```
mouse_links/
├── manifest.json                    # Extension manifest (includes nativeMessaging permission)
├── background.js                    # Handles native messaging connection
├── content.js                       # Clicks links based on color
├── native_host.py                   # Native messaging host (Python)
├── com.closest_links.host.json      # Native host manifest (template)
├── install_native_host.sh           # Installation script
└── send_link_command.py             # Test Python application
```

## Troubleshooting

### Error: "Specified native messaging host not found"

**Cause:** The native host manifest isn't installed in the correct location or has the wrong Extension ID.

**Solution:**
1. Run `bash install_native_host.sh` again
2. Make sure you entered the correct Extension ID (32 characters)
3. Verify the manifest is at: `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.closest_links.host.json`
4. Check the manifest contains your correct Extension ID

### Error: "Native messaging host has exited"

**Cause:** The native_host.py script has an error or can't be executed.

**Solution:**
1. Make sure `native_host.py` is executable:
   ```bash
   chmod +x native_host.py
   ```
2. Test the script manually:
   ```bash
   python3 native_host.py
   ```
3. Check logs at `~/closest_links_host.log`

### No response when running send_link_command.py

**Causes:**
- Extension not activated on the current page
- No links highlighted (mouse hasn't moved)
- Link with that color doesn't exist

**Solutions:**
1. Make sure extension is activated (`Cmd+Shift+L`)
2. Move mouse to highlight links
3. Check browser console (F12) for errors
4. Check native host logs: `~/closest_links_host.log`

### "Permission denied" when running scripts

**Solution:**
```bash
chmod +x native_host.py
chmod +x install_native_host.sh
chmod +x send_link_command.py
```

## Debugging

### Check Native Host Logs

The native host writes detailed logs to `~/closest_links_host.log`:

```bash
tail -f ~/closest_links_host.log
```

**Expected log entries:**
```
[INIT] Native messaging host starting
[INIT] Socket server thread started
[SOCKET] Server listening on localhost:9999
[→CHROME] {'status': 'connected', 'host': 'com.closest_links.host'}
[SOCKET] Client connected: ('127.0.0.1', 54321)
[SOCKET→] Received: {'command': 'click-link', 'number': 2}
[→CHROME] {'command': 'click-link', 'number': 2}
```

### Check Extension Console

1. Go to `chrome://extensions/`
2. Click **"Inspect views: service worker"**
3. Look for connection status and any error messages

### Test Socket Connection

You can test the socket connection directly:

```bash
# In terminal 1: Monitor native host logs
tail -f ~/closest_links_host.log

# In terminal 2: Test socket connection
python3 -c "
import socket, json
sock = socket.socket()
sock.connect(('localhost', 9999))
sock.send(json.dumps({'command': 'click-link', 'number': 1}).encode())
print('Response:', sock.recv(1024).decode())
sock.close()
"
```

You should see the message logged in the native host logs.

## Advanced Usage

### Using in Your Own Python Application

You can integrate the link clicking functionality into your own Python application:

```python
import json
import struct
import subprocess

def click_link(number):
    """Click a highlighted link by number (1, 2, or 3)"""
    message = {
        'command': 'click-link',
        'number': number
    }
    
    # Path to native host
    native_host = '/path/to/native_host.py'
    
    # Start native host
    process = subprocess.Popen(
        ['python3', native_host],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    
    # Send message
    encoded = json.dumps(message).encode('utf-8')
    process.stdin.write(struct.pack('I', len(encoded)))
    process.stdin.write(encoded)
    process.stdin.flush()
    
    # Close
    process.stdin.close()
    process.terminate()

# Example usage
click_link(1)  # Click red link
```

### Extending the Protocol

You can add more commands by modifying:

1. **content.js** - Add new command handlers
2. **background.js** - Forward new commands
3. **native_host.py** - Process new message types
4. Your Python app - Send new commands

Example new command:

```python
# In send_link_command.py
message = {
    'command': 'get-link-info',
    'number': 1
}
```

## Security Considerations

- Native messaging requires explicit configuration (Extension ID in manifest)
- Only the specified extension can communicate with the native host
- The native host runs with your user's permissions
- Messages are local-only (no network communication)

## Uninstallation

To remove the native messaging host:

```bash
rm ~/Library/Application\ Support/Google/Chrome/NativeMessagingHosts/com.closest_links.host.json
rm ~/closest_links_host.log
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs at `~/closest_links_host.log`
3. Check the extension service worker console
4. Ensure all files have correct permissions

## References

- [Chrome Native Messaging Documentation](https://developer.chrome.com/docs/apps/nativeMessaging/)
- [Native Messaging Protocol](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging#native-messaging-host-protocol)
