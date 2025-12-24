// Chrome Extension: Closest Links Highlighter
// Finds and highlights the 3 closest links to mouse pointer

(function() {
  'use strict';

  // Configuration
  const COLORS = ['#FF6666', '#66DD66', '#66BBFF']; // Bright Red, Green, Blue
  const THROTTLE_DELAY = 50; // ms
  const BORDER_WIDTH = '2px';
  const BORDER_STYLE = 'solid';

  // State
  let isActive = true;  // Default value (will be loaded from storage)
  let mouseX = 0;
  let mouseY = 0;
  let linkColorMap = new Map(); // Map<HTMLElement, string>
  let availableColors = [...COLORS];
  let currentTopLinks = new Set(); // Set<HTMLElement>
  let isInitialized = false;

  // Throttle function
  function throttle(func, delay) {
    let lastCall = 0;
    let timeoutId = null;

    return function(...args) {
      const now = Date.now();
      const timeSinceLastCall = now - lastCall;

      if (timeSinceLastCall >= delay) {
        lastCall = now;
        func.apply(this, args);
      } else {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          lastCall = Date.now();
          func.apply(this, args);
        }, delay - timeSinceLastCall);
      }
    };
  }

  // Calculate distance from point to rectangle
  function distanceToRect(x, y, rect) {
    // If point is inside the rectangle, distance is 0
    if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
      return 0;
    }
    
    // Otherwise, calculate distance to nearest edge
    const dx = Math.max(rect.left - x, 0, x - rect.right);
    const dy = Math.max(rect.top - y, 0, y - rect.bottom);
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Check if element is visible
  function isElementVisible(el) {
    // Check if element has dimensions
    if (el.offsetWidth === 0 || el.offsetHeight === 0) {
      return false;
    }
    
    // Check computed styles
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
      return false;
    }
    
    // Check if element is in the DOM
    if (!document.body.contains(el)) {
      return false;
    }
    
    return true;
  }

  // Get all clickable elements on the page (links, buttons, etc.)
  function getAllLinks() {
    const clickables = new Set();
    
    // Get all <a> tags with href
    document.querySelectorAll('a[href]').forEach(el => {
      if (isElementVisible(el)) clickables.add(el);
    });
    
    // Get all <button> elements
    document.querySelectorAll('button').forEach(el => {
      if (isElementVisible(el)) clickables.add(el);
    });
    
    // Get elements with role="button"
    document.querySelectorAll('[role="button"]').forEach(el => {
      if (isElementVisible(el)) clickables.add(el);
    });
    
    // Get elements with onclick attribute
    document.querySelectorAll('[onclick]').forEach(el => {
      if (isElementVisible(el)) clickables.add(el);
    });
    
    // Get elements with cursor: pointer style
    document.querySelectorAll('*').forEach(el => {
      const style = window.getComputedStyle(el);
      if (style.cursor === 'pointer' && isElementVisible(el)) {
        clickables.add(el);
      }
    });
    
    return Array.from(clickables);
  }

  // Check if link is in viewport
  function isInViewport(rect) {
    return (
      rect.top < window.innerHeight &&
      rect.bottom > 0 &&
      rect.left < window.innerWidth &&
      rect.right > 0
    );
  }

  // Find the 3 closest links to mouse pointer
  function findClosestLinks(x, y) {
    const links = getAllLinks();
    const linkDistances = [];

    for (const link of links) {
      const rect = link.getBoundingClientRect();
      
      // Skip links with zero dimensions
      if (rect.width === 0 || rect.height === 0) continue;

      const distance = distanceToRect(x, y, rect);
      linkDistances.push({ link, distance, rect });
    }

    // Sort by distance and get top 3
    linkDistances.sort((a, b) => a.distance - b.distance);
    return linkDistances.slice(0, 3).map(item => item.link);
  }

  // Assign color to a link, reusing existing color if available
  function assignColor(link) {
    // If link already has a color assigned, keep it
    if (linkColorMap.has(link)) {
      return linkColorMap.get(link);
    }

    // Get an available color
    let color;
    if (availableColors.length > 0) {
      color = availableColors.shift();
    } else {
      // Fallback: use first color if all are taken (shouldn't happen with 3 colors and 3 links)
      color = COLORS[0];
    }

    linkColorMap.set(link, color);
    return color;
  }

  // Free color from a link that's no longer in top 3
  function freeColor(link) {
    const color = linkColorMap.get(link);
    if (color) {
      linkColorMap.delete(link);
      // Only add back to available if it's not currently used by any other link
      const colorInUse = Array.from(linkColorMap.values()).includes(color);
      if (!colorInUse && !availableColors.includes(color)) {
        availableColors.push(color);
      }
    }
  }

  // Highlight the closest links
  function highlightClosestLinks() {
    const closestLinks = findClosestLinks(mouseX, mouseY);
    const newTopLinks = new Set(closestLinks);

    // Check which links are in viewport
    const viewportLinks = closestLinks.filter(link => {
      const rect = link.getBoundingClientRect();
      return isInViewport(rect);
    });

    // Remove highlighting from links no longer in top 3
    for (const link of currentTopLinks) {
      if (!newTopLinks.has(link)) {
        link.style.outline = '';
        freeColor(link);
      }
    }

    // Add highlighting to current top 3 links
    for (const link of closestLinks) {
      const rect = link.getBoundingClientRect();
      
      // Only highlight if in viewport
      if (isInViewport(rect)) {
        const color = assignColor(link);
        link.style.outline = `${BORDER_WIDTH} ${BORDER_STYLE} ${color}`;
        link.style.outlineOffset = '2px';
      }
    }

    currentTopLinks = newTopLinks;
  }

  // Clear all highlights
  function clearAllHighlights() {
    const links = getAllLinks();
    for (const link of links) {
      link.style.outline = '';
      link.style.outlineOffset = '';
    }
    linkColorMap.clear();
    availableColors = [...COLORS];
    currentTopLinks.clear();
  }

  // Handle mouse move
  const handleMouseMove = throttle((event) => {
    if (!isActive) return;
    
    mouseX = event.clientX;
    mouseY = event.clientY;
    highlightClosestLinks();
  }, THROTTLE_DELAY);

  // Save activation state to storage
  function saveActivationState(active) {
    chrome.storage.sync.set({ isActive: active }, () => {
      console.log('[STORAGE] Saved state:', active);
    });
  }

  // Load activation state from storage
  function loadActivationState(callback) {
    chrome.storage.sync.get(['isActive'], (result) => {
      const active = result.isActive !== undefined ? result.isActive : true;
      console.log('[STORAGE] Loaded state:', active);
      callback(active);
    });
  }

  // Toggle activation
  function toggleActivation() {
    isActive = !isActive;
    
    // Save to storage for persistence across all tabs
    saveActivationState(isActive);
    
    if (isActive) {
      console.log('Closest Links Highlighter: ACTIVATED (saved globally)');
      // Start tracking immediately if mouse has moved
      if (mouseX !== 0 || mouseY !== 0) {
        highlightClosestLinks();
      }
    } else {
      console.log('Closest Links Highlighter: DEACTIVATED (saved globally)');
      clearAllHighlights();
    }
  }

  // Apply activation state
  function applyActivationState(active) {
    isActive = active;
    isInitialized = true;
    
    if (isActive) {
      console.log('[INIT] Extension auto-activated');
      // Start highlighting if mouse has moved
      if (mouseX !== 0 || mouseY !== 0) {
        highlightClosestLinks();
      }
    } else {
      console.log('[INIT] Extension deactivated (persisted from previous session)');
      clearAllHighlights();
    }
  }

  // Map numbers to colors
  const NUMBER_TO_COLOR = {
    1: '#FF6666', // Bright Red
    2: '#66DD66', // Bright Green
    3: '#66BBFF'  // Bright Blue
  };

  // Click link by color
  function clickLinkByColor(color) {
    console.log('[CLICK] Target color:', color, '| Map size:', linkColorMap.size);
    
    // Find the link that has this color assigned
    for (const [link, assignedColor] of linkColorMap.entries()) {
      if (assignedColor === color) {
        console.log('[CLICK] ✓ Found:', link.href);
        link.click();
        return true;
      }
    }
    
    console.warn('[CLICK] ✗ Not found. Available:', Array.from(linkColorMap.values()));
    return false;
  }

  // Listen for messages via Chrome API
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('[MSG]', request.command, JSON.stringify(request));
    
    if (request.command === 'toggle-highlight') {
      toggleActivation();
      sendResponse({ status: 'ok', active: isActive });
    } else if (request.command === 'click-link' && request.number) {
      const number = parseInt(request.number);
      if (number >= 1 && number <= 3) {
        const color = NUMBER_TO_COLOR[number];
        console.log('[EXEC] Number:', number, '→ Color:', color, '| Active:', isActive);
        const success = clickLinkByColor(color);
        const response = { status: success ? 'ok' : 'not_found', number, color, active: isActive };
        console.log('[RESP]', JSON.stringify(response));
        sendResponse(response);
      } else {
        sendResponse({ status: 'invalid_number', number: number });
      }
    }
    return true;
  });

  // Initialize mouse tracking
  document.addEventListener('mousemove', handleMouseMove);

  // Handle viewport changes
  let resizeTimeout;
  window.addEventListener('resize', () => {
    if (!isActive) return;
    
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      highlightClosestLinks();
    }, 100);
  });

  // Handle scroll events
  let scrollTimeout;
  window.addEventListener('scroll', () => {
    if (!isActive) return;
    
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      highlightClosestLinks();
    }, 50);
  });

  // Handle DOM mutations (links added/removed)
  const observer = new MutationObserver(() => {
    if (!isActive) return;
    highlightClosestLinks();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  // Load activation state from storage and initialize
  console.log('[INIT] Closest Links Highlighter | Loading global state...');
  loadActivationState((active) => {
    applyActivationState(active);
    console.log('[INIT] State loaded | Shortcut: Cmd+Shift+L to toggle | Colors: 1=Red 2=Green 3=Blue');
  });

  // Listen for storage changes from other tabs
  chrome.storage.onChanged.addListener((changes, namespace) => {
    if (namespace === 'sync' && changes.isActive) {
      const newState = changes.isActive.newValue;
      console.log('[STORAGE] State changed in another tab:', newState);
      isActive = newState;
      
      if (isActive) {
        console.log('[SYNC] Activated from another tab');
        if (mouseX !== 0 || mouseY !== 0) {
          highlightClosestLinks();
        }
      } else {
        console.log('[SYNC] Deactivated from another tab');
        clearAllHighlights();
      }
    }
  });
})();
