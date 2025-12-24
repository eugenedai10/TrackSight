#!/usr/bin/env python3
"""
UART Reader Module

Reads binary state (0 or 1) from a UART device in a separate thread.
The state is stored in a global variable for use by other modules.
"""

import serial
import threading
import logging
import time

# Global UART state: 0=OFF, 1=ON
# Using a list so it's mutable and shared across imports
uart_control_state = [0]  # Access with uart_control_state[0]

# Global push button states (bits 1-3 from UART data)
# Using a list so it's mutable and shared across imports
uart_button_states = [0, 0, 0]  # button1, button2, button3


class UARTReader:
    """Reads from UART device in a separate thread."""
    
    def __init__(self, port="/dev/tty.usbmodem141301", baudrate=115200):
        """Initialize UART reader.
        
        Args:
            port: Serial port path (e.g., /dev/tty.usbmodem21201)
            baudrate: Communication speed (default: 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.thread = None
        self.serial_conn = None
        
    def start(self):
        """Start the UART reading thread."""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logging.info(f"UART reader started on {self.port} at {self.baudrate} baud")
        
    def stop(self):
        """Stop the UART reading thread."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                logging.info("UART connection closed")
            except Exception as e:
                logging.warning(f"Error closing UART: {e}")
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _read_loop(self):
        """Main loop to read from UART device."""
        global uart_control_state, uart_button_states
        
        while self.running:
            try:
                # Open serial connection if not already open
                if not self.serial_conn or not self.serial_conn.is_open:
                    self.serial_conn = serial.Serial(
                        self.port, 
                        self.baudrate, 
                        timeout=1
                    )
                    logging.info(f"UART connected: {self.port}")
                
                # Read line from serial port
                line = self.serial_conn.readline().decode('utf-8').strip()
                
                # Parse the integer value from the line
                value = int(line)
                
                # Extract bit 0 for control state
                uart_control_state[0] = value & 0x01
                
                # Extract bits 1-3 for push button states
                uart_button_states[0] = (value >> 1) & 0x01  # Bit 1
                uart_button_states[1] = (value >> 2) & 0x01  # Bit 2
                uart_button_states[2] = (value >> 3) & 0x01  # Bit 3
                
                # logging.info(f"UART read: {value} => control={uart_control_state[0]}, buttons={uart_button_states}")
                    
            except serial.SerialException as e:
                logging.warning(f"UART connection error: {e}")
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                    except:
                        pass
                time.sleep(1)  # Wait before retry
            except Exception as e:
                logging.warning(f"UART read error: {e}")
                time.sleep(1)  # Wait before retry
