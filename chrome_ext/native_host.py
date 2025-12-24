#!/usr/bin/env python3
"""
Native Messaging Host for Closest Links Highlighter Chrome Extension
Communicates with Chrome via stdin/stdout and listens for external commands via socket.
"""

import sys
import json
import struct
import logging
import socket
import threading
from pathlib import Path

# Configuration
SOCKET_PORT = 9999
SOCKET_HOST = 'localhost'

# Set up logging
log_file = Path.home() / 'closest_links_host.log'
logging.basicConfig(
    filename=str(log_file),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def send_message_to_chrome(message):
    """Send a message to Chrome extension using native messaging protocol."""
    try:
        encoded_message = json.dumps(message).encode('utf-8')
        message_length = len(encoded_message)
        
        sys.stdout.buffer.write(struct.pack('I', message_length))
        sys.stdout.buffer.write(encoded_message)
        sys.stdout.buffer.flush()
        
        logging.info(f'[→CHROME] {message}')
    except Exception as e:
        logging.error(f'[→CHROME] Error: {e}')

def read_message_from_chrome():
    """Read a message from Chrome extension."""
    try:
        raw_length = sys.stdin.buffer.read(4)
        if len(raw_length) == 0:
            return None
        
        message_length = struct.unpack('I', raw_length)[0]
        message = sys.stdin.buffer.read(message_length).decode('utf-8')
        data = json.loads(message)
        
        logging.info(f'[CHROME→] {data}')
        return data
    except Exception as e:
        logging.error(f'[CHROME→] Error: {e}')
        return None

def handle_socket_client(client_socket, addr):
    """Handle incoming socket connection from external app."""
    try:
        logging.info(f'[SOCKET] Client connected: {addr}')
        
        # Receive data from external app
        data = client_socket.recv(1024).decode('utf-8')
        if data:
            message = json.loads(data)
            logging.info(f'[SOCKET→] Received: {message}')
            
            # Forward to Chrome extension
            send_message_to_chrome(message)
            
            # Send acknowledgment back to client
            response = {'status': 'forwarded'}
            client_socket.send(json.dumps(response).encode('utf-8'))
            logging.info(f'[→SOCKET] Sent: {response}')
    except Exception as e:
        logging.error(f'[SOCKET] Error: {e}')
    finally:
        client_socket.close()
        logging.info(f'[SOCKET] Client disconnected: {addr}')

def socket_server():
    """Run socket server to receive commands from external apps."""
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((SOCKET_HOST, SOCKET_PORT))
        server.listen(5)
        
        logging.info(f'[SOCKET] Server listening on {SOCKET_HOST}:{SOCKET_PORT}')
        
        while True:
            client, addr = server.accept()
            client_thread = threading.Thread(target=handle_socket_client, args=(client, addr))
            client_thread.daemon = True
            client_thread.start()
    except Exception as e:
        logging.error(f'[SOCKET] Server error: {e}')

def chrome_message_loop():
    """Read messages from Chrome extension."""
    try:
        while True:
            message = read_message_from_chrome()
            if message is None:
                break
            
            # Log response from Chrome (if needed)
            logging.info(f'[CHROME] Response: {message}')
    except Exception as e:
        logging.error(f'[CHROME] Loop error: {e}')

def main():
    """Main entry point."""
    logging.info('=' * 50)
    logging.info('[INIT] Native messaging host starting')
    logging.info('=' * 50)
    
    try:
        # Start socket server in background thread
        socket_thread = threading.Thread(target=socket_server)
        socket_thread.daemon = True
        socket_thread.start()
        logging.info('[INIT] Socket server thread started')
        
        # Send initial connection confirmation to Chrome
        send_message_to_chrome({'status': 'connected', 'host': 'com.closest_links.host'})
        
        # Run Chrome message loop in main thread
        chrome_message_loop()
        
    except KeyboardInterrupt:
        logging.info('[EXIT] Host interrupted by user')
    except Exception as e:
        logging.error(f'[EXIT] Unexpected error: {e}')
    finally:
        logging.info('[EXIT] Native messaging host stopped')

if __name__ == '__main__':
    main()
