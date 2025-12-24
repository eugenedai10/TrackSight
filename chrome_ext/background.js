// Background service worker for handling keyboard shortcuts and native messaging

// Native messaging port
let nativePort = null;
const NATIVE_HOST_NAME = 'com.closest_links.host';

// Connect to native messaging host
function connectNativeHost() {
  try {
    nativePort = chrome.runtime.connectNative(NATIVE_HOST_NAME);
    
    nativePort.onMessage.addListener((message) => {
      console.log('[NATIVE→BG]', JSON.stringify(message));
      
      if (message.command === 'click-link' && message.number) {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (tabs[0]) {
            console.log('[BG→TAB]', tabs[0].id, JSON.stringify({command: message.command, number: message.number}));
            
            chrome.tabs.sendMessage(tabs[0].id, {command: 'click-link', number: message.number}, (response) => {
              if (chrome.runtime.lastError) {
                console.error('[BG] Error:', chrome.runtime.lastError.message);
                if (nativePort) {
                  nativePort.postMessage({status: 'error', error: chrome.runtime.lastError.message});
                }
              } else if (response) {
                console.log('[TAB→BG]', JSON.stringify(response));
                if (nativePort) {
                  console.log('[BG→NATIVE]', JSON.stringify(response));
                  nativePort.postMessage(response);
                }
              }
            });
          } else {
            console.error('[BG] No active tab');
            if (nativePort) {
              nativePort.postMessage({status: 'error', error: 'No active tab'});
            }
          }
        });
      }
    });
    
    nativePort.onDisconnect.addListener(() => {
      console.log('[NATIVE] Disconnected');
      if (chrome.runtime.lastError) {
        console.error('[NATIVE] Error:', chrome.runtime.lastError.message);
      }
      nativePort = null;
    });
    
    console.log('[NATIVE] ✓ Connected:', NATIVE_HOST_NAME);
  } catch (error) {
    console.error('[NATIVE] ✗ Connection failed:', error);
    nativePort = null;
  }
}

// Try to connect on startup
console.log('[BG] Starting native host connection...');
connectNativeHost();

// Handle keyboard shortcuts
chrome.commands.onCommand.addListener((command) => {
  if (command === 'toggle-highlight') {
    // Send message to active tab's content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { command: 'toggle-highlight' }, (response) => {
          if (chrome.runtime.lastError) {
            console.log('Could not send message:', chrome.runtime.lastError.message);
          } else if (response) {
            console.log('Toggle status:', response.active ? 'ACTIVE' : 'INACTIVE');
          }
        });
      }
    });
  }
});

// Listen for messages from content script or other parts of extension
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.command === 'reconnect-native-host') {
    connectNativeHost();
    sendResponse({ status: 'ok' });
  }
  return true;
});
