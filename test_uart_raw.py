#!/usr/bin/env python3
"""
Simple Raw UART Reader

Reads and displays raw data from UART device.
Press Ctrl+C to exit.
"""

import serial
import time
import sys

def main():
    """Main test function."""
    print("=" * 60)
    print("Raw UART Reader Test")
    print("=" * 60)
    print()
    
    # Get UART port from command line or use default
    uart_port = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem21201"
    baudrate = 115200
    
    print(f"UART Port: {uart_port}")
    print(f"Baud Rate: {baudrate}")
    print()
    print("Press Ctrl+C to exit")
    print("-" * 60)
    print()
    
    try:
        # Open serial port
        ser = serial.Serial(uart_port, baudrate, timeout=1)
        print(f"✓ Connected to {uart_port}")
        print(f"✓ Port is open: {ser.is_open}")
        print()
        print("Waiting for data... (send '0' or '1' from your device)")
        print()
        
        last_status_time = time.time()
        data_received = False
        
        while True:
            # Show periodic status if no data received
            if not data_received and time.time() - last_status_time > 3:
                print(f"[{time.strftime('%H:%M:%S')}] Still waiting for data... (bytes in buffer: {ser.in_waiting})")
                last_status_time = time.time()
            
            # Read a line from serial
            if ser.in_waiting > 0:
                raw_data = ser.readline()
                data_received = True
                
                # Display raw bytes
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] Raw bytes: {raw_data}")
                
                # Try to decode as string
                try:
                    decoded = raw_data.decode('utf-8').strip()
                    print(f"[{timestamp}] Decoded:   '{decoded}'")
                    print(f"[{timestamp}] Length:    {len(decoded)} chars")
                except Exception as e:
                    print(f"[{timestamp}] Decoded:   <decode error: {e}>")
                
                print()
            
            time.sleep(0.01)  # Small delay
            
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check if the device is connected")
        print("2. Verify the port name (use 'ls /dev/tty.*' on macOS)")
        print("3. Make sure no other program is using the port")
    except KeyboardInterrupt:
        print()
        print("-" * 60)
        print("Test interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")
        print("Test completed")


if __name__ == "__main__":
    main()
