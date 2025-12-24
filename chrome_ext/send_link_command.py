#!/usr/bin/env python3
"""
Test application to send link click commands to the Chrome extension.
Connects to the native messaging host via socket.

Usage:
    python3 send_link_command.py 1    # Click red link
    python3 send_link_command.py 2    # Click green link
    python3 send_link_command.py 3    # Click blue link
"""

import sys
import json
import socket

# Socket configuration (must match native_host.py)
SOCKET_HOST = 'localhost'
SOCKET_PORT = 9999

def send_command(message):
    """Send a command to the native host via socket."""
    try:
        # Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        
        print(f"Connecting to native host at {SOCKET_HOST}:{SOCKET_PORT}...")
        sock.connect((SOCKET_HOST, SOCKET_PORT))
        print("✓ Connected")
        
        # Send message
        message_json = json.dumps(message)
        sock.send(message_json.encode('utf-8'))
        print(f"✓ Sent: {message_json}")
        
        # Receive response
        response = sock.recv(1024).decode('utf-8')
        if response:
            response_data = json.loads(response)
            print(f"✓ Response: {json.dumps(response_data)}")
        
        sock.close()
        return True
        
    except ConnectionRefusedError:
        print("\n✗ ERROR: Could not connect to native host")
        print("\nTroubleshooting:")
        print("1. Make sure the Chrome extension is loaded")
        print("2. Check if the extension service worker is running:")
        print("   - Go to chrome://extensions/")
        print("   - Click 'Inspect views: service worker'")
        print("3. Check native host logs: tail -f ~/closest_links_host.log")
        print("4. The native host should show: [SOCKET] Server listening on localhost:9999")
        return False
        
    except socket.timeout:
        print("\n✗ ERROR: Connection timeout")
        print("The native host may not be responding.")
        return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

def main():
    """Main function to parse arguments and send commands."""
    if len(sys.argv) != 2:
        print("Usage: python3 send_link_command.py <number>")
        print("")
        print("  <number>  Link to click (1, 2, or 3)")
        print("            1 = Red link")
        print("            2 = Green link")
        print("            3 = Blue link")
        print("")
        print("Example:")
        print("  python3 send_link_command.py 1")
        sys.exit(1)
    
    try:
        number = int(sys.argv[1])
        
        if number < 1 or number > 3:
            print("✗ ERROR: Number must be 1, 2, or 3")
            sys.exit(1)
        
        color_map = {1: 'Red', 2: 'Green', 3: 'Blue'}
        print(f"Sending command to click {color_map[number]} link (#{number})...\n")
        
        # Create the message
        message = {
            'command': 'click-link',
            'number': number
        }
        
        # Send the command
        success = send_command(message)
        
        if success:
            print("\n✓ Command sent successfully!")
            print("\nMake sure:")
            print("1. A webpage is open with links")
            print("2. Extension is activated (Cmd+Shift+L)")
            print("3. Links are highlighted (move your mouse)")
            print("\nCheck browser console (F12) to see the click action.")
        else:
            sys.exit(1)
            
    except ValueError:
        print("✗ ERROR: Argument must be a number (1, 2, or 3)")
        sys.exit(1)

if __name__ == '__main__':
    main()
