#!/usr/bin/env python3
"""
Detect and display all screen configurations on macOS
"""

from AppKit import NSScreen

def detect_screens():
    """Detect all connected screens and their configurations"""
    
    screens = NSScreen.screens()
    print(f"\n{'='*70}")
    print(f"Detected {len(screens)} screen(s)")
    print(f"{'='*70}\n")
    
    for idx, screen in enumerate(screens):
        frame = screen.frame()
        visible_frame = screen.visibleFrame()
        
        # Screen origin and size
        origin_x = frame.origin.x
        origin_y = frame.origin.y
        width = frame.size.width
        height = frame.size.height
        
        # Visible frame (excluding menu bar, dock, etc.)
        visible_x = visible_frame.origin.x
        visible_y = visible_frame.origin.y
        visible_width = visible_frame.size.width
        visible_height = visible_frame.size.height
        
        # Device description
        description = screen.deviceDescription()
        
        # Check if this is the main screen (with menu bar)
        is_main = screen == NSScreen.mainScreen()
        
        print(f"Screen {idx}:")
        print(f"  {'[MAIN SCREEN]' if is_main else '[Secondary]'}")
        print(f"  Frame:")
        print(f"    Origin: ({origin_x:.0f}, {origin_y:.0f})")
        print(f"    Size:   {width:.0f} × {height:.0f}")
        print(f"    Bounds: X:[{origin_x:.0f} to {origin_x + width:.0f}], "
              f"Y:[{origin_y:.0f} to {origin_y + height:.0f}]")
        print(f"  Visible Frame (excluding menu bar/dock):")
        print(f"    Origin: ({visible_x:.0f}, {visible_y:.0f})")
        print(f"    Size:   {visible_width:.0f} × {visible_height:.0f}")
        
        # Extract some useful device info if available
        if 'NSScreenNumber' in description:
            screen_number = description['NSScreenNumber']
            print(f"  Screen Number: {screen_number}")
        
        # Display backing scale factor (Retina vs non-Retina)
        backing_scale = screen.backingScaleFactor()
        print(f"  Backing Scale Factor: {backing_scale}x "
              f"({'Retina' if backing_scale > 1 else 'Non-Retina'})")
        
        print()
    
    # Show ASCII layout visualization
    print(f"{'='*70}")
    print("Screen Layout Visualization:")
    print(f"{'='*70}\n")
    visualize_screen_layout(screens)
    
    return screens

def visualize_screen_layout(screens):
    """Create ASCII art visualization of screen layout"""
    
    # Find bounds of all screens
    min_x = min(screen.frame().origin.x for screen in screens)
    max_x = max(screen.frame().origin.x + screen.frame().size.width for screen in screens)
    min_y = min(screen.frame().origin.y for screen in screens)
    max_y = max(screen.frame().origin.y + screen.frame().size.height for screen in screens)
    
    total_width = max_x - min_x
    total_height = max_y - min_y
    
    print(f"Total Virtual Desktop: {total_width:.0f} × {total_height:.0f}")
    print(f"Bounds: X:[{min_x:.0f} to {max_x:.0f}], Y:[{min_y:.0f} to {max_y:.0f}]")
    print()
    
    # Simple text representation
    for idx, screen in enumerate(screens):
        frame = screen.frame()
        x = frame.origin.x
        y = frame.origin.y
        w = frame.size.width
        h = frame.size.height
        
        is_main = screen == NSScreen.mainScreen()
        marker = "MAIN" if is_main else f"S{idx}"
        
        print(f"Screen {idx} ({marker}): "
              f"[{x:.0f}, {y:.0f}] to [{x+w:.0f}, {y+h:.0f}]  "
              f"({w:.0f}×{h:.0f})")
    
    print("\nNote: Coordinates use macOS Cocoa system (origin at bottom-left)")

def convert_coordinates_example(screens):
    """Show coordinate conversion examples"""
    
    print(f"\n{'='*70}")
    print("Coordinate System Conversion Examples:")
    print(f"{'='*70}\n")
    
    main_screen = NSScreen.mainScreen()
    main_height = main_screen.frame().size.height
    
    print(f"Main screen height: {main_height:.0f}")
    print()
    
    # Example points in Cocoa coordinates
    example_points = [
        (0, 0, "Bottom-left of primary screen (Cocoa origin)"),
        (0, main_height, "Top-left of primary screen"),
        (100, 100, "Near bottom-left (Cocoa)"),
        (100, main_height - 100, "Near top-left"),
    ]
    
    print("Cocoa (macOS) → Standard (Top-Left Origin)")
    print("-" * 70)
    for cocoa_x, cocoa_y, desc in example_points:
        standard_y = main_height - cocoa_y
        print(f"Cocoa: ({cocoa_x:6.0f}, {cocoa_y:6.0f}) → "
              f"Standard: ({cocoa_x:6.0f}, {standard_y:6.0f})  # {desc}")

if __name__ == "__main__":
    print("\nMacOS Screen Detection")
    screens = detect_screens()
    convert_coordinates_example(screens)
    print()
