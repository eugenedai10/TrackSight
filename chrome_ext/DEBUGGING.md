# Debugging Guide for Closest Links Highlighter

## Step-by-Step Debugging Process

### Step 1: Verify Extension Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Ensure "Developer mode" is enabled (toggle in top-right)
3. Check that "Closest Links Highlighter" appears in the list
4. Verify the extension is **enabled** (blue toggle switch)
5. Click "Details" and check for any errors in the "Errors" section

**Expected Result:** Extension should be listed and enabled with no errors.

---

### Step 2: Check Extension Console for Errors

1. Go to `chrome://extensions/`
2. Find "Closest Links Highlighter"
3. Click "**Inspect views: service worker**" (this opens the background script console)
4. Look for any error messages in red

**What to look for:**
- Syntax errors
- Registration errors
- Any other red error messages

---

### Step 3: Test on a Simple Page

1. Open `test.html` in Chrome (or any webpage with links)
2. **Right-click** on the page → Select "**Inspect**" (or press F12)
3. Go to the **Console** tab
4. Look for the message: `"Closest Links Highlighter extension loaded..."`

**Expected Result:** You should see the extension loaded message in the console.

**If you DON'T see this message:**
- The content script isn't loading properly
- Try refreshing the page (Ctrl+R or Cmd+R)
- Check if the extension is enabled

---

### Step 4: Test the Keyboard Shortcut

1. With the test page open and console visible
2. Press **Ctrl+Shift+L** (Windows/Linux) or **Cmd+Shift+L** (Mac)
3. Look in the console for: `"Closest Links Highlighter: ACTIVATED"`

**Expected Result:** You should see the activation message.

**If the shortcut doesn't work:**

#### Option A: Check Keyboard Shortcut Configuration
1. Go to `chrome://extensions/shortcuts`
2. Find "Closest Links Highlighter"
3. Verify the keyboard shortcut is set to `Ctrl+Shift+L` (or `Cmd+Shift+L`)
4. If not set or conflicts with another extension, click the pencil icon to change it

#### Option B: Manual Activation via Console
1. Open the page's console (F12 → Console tab)
2. Type this command and press Enter:
   ```javascript
   chrome.runtime.sendMessage({command: 'toggle-highlight'})
   ```
3. If this doesn't work, try directly toggling in the content script:
   ```javascript
   // Check if the script variables exist
   console.log('Testing extension...');
   ```

---

### Step 5: Test Link Detection

If the extension activates but doesn't highlight links:

1. Open console (F12)
2. After activating the extension, run this in console:
   ```javascript
   // Check how many links are detected
   console.log('Total links found:', document.querySelectorAll('a[href]').length);
   ```

**Expected Result:** Should show a number > 0

---

### Step 6: Manual Test in Console

Run this code in the browser console to test if the core functionality works:

```javascript
// Test distance calculation
function testHighlighting() {
  const links = document.querySelectorAll('a[href]');
  console.log('Found', links.length, 'links');
  
  // Manually highlight first 3 links
  links.forEach((link, index) => {
    if (index < 3) {
      const colors = ['#FF0000', '#00FF00', '#0000FF'];
      link.style.outline = `3px solid ${colors[index]}`;
      link.style.outlineOffset = '2px';
      console.log('Highlighted link', index, ':', link.textContent);
    }
  });
}

testHighlighting();
```

**Expected Result:** First 3 links on the page should be highlighted in red, green, blue.

---

## Common Issues & Solutions

### Issue 1: Extension Not Loading
**Symptoms:** No console message about extension loaded
**Solutions:**
- Reload the extension: Go to `chrome://extensions/`, find your extension, click the reload icon (circular arrow)
- Refresh the test page after reloading extension
- Check if file permissions are correct (extension folder should be readable)

### Issue 2: Keyboard Shortcut Not Working
**Symptoms:** No activation message when pressing shortcut
**Solutions:**
- Check `chrome://extensions/shortcuts` for conflicts
- Try a different shortcut (e.g., `Alt+Shift+L`)
- Some operating systems or other software may intercept certain shortcuts
- Use the manual activation in console as shown in Step 4, Option B

### Issue 3: Links Not Highlighting
**Symptoms:** Extension activates but no visual changes
**Solutions:**
- Check if page has actual `<a href="...">` links
- Verify you're moving the mouse (highlighting is real-time)
- Check console for JavaScript errors
- Try the manual test in Step 6

### Issue 4: Service Worker Issues (Manifest V3)
**Symptoms:** Background script console shows errors
**Solutions:**
- Ensure Chrome version is 88+ (Manifest V3 requirement)
- Reload the extension completely
- Check for syntax errors in `background.js`

---

## Alternative: Simplified Testing Method

If you want to quickly test if the core logic works, create this simple test file:

### quick-test.html
```html
<!DOCTYPE html>
<html>
<head><title>Quick Test</title></head>
<body style="padding: 50px;">
    <h1>Quick Test Page</h1>
    <p>Move your mouse around after activating!</p>
    <div style="margin: 50px 0;">
        <a href="#1" style="margin: 20px; display: inline-block;">Link 1</a>
        <a href="#2" style="margin: 20px; display: inline-block;">Link 2</a>
        <a href="#3" style="margin: 20px; display: inline-block;">Link 3</a>
        <a href="#4" style="margin: 20px; display: inline-block;">Link 4</a>
        <a href="#5" style="margin: 20px; display: inline-block;">Link 5</a>
    </div>
    <p>Press Ctrl+Shift+L to activate, then move your mouse!</p>
</body>
</html>
```

---

## What Information to Provide

If you're still having issues, please share:

1. **Chrome version**: Help → About Google Chrome
2. **Operating System**: Windows, Mac, or Linux?
3. **Console output**: Copy any error messages from both:
   - Service worker console (`chrome://extensions/` → Inspect views)
   - Page console (F12 → Console)
4. **Extension status**: Is it enabled? Any errors in `chrome://extensions/`?
5. **Keyboard shortcut test**: Does going to `chrome://extensions/shortcuts` show the shortcut correctly?

---

## Quick Diagnostic Command

Run this in the console of any page to diagnose:

```javascript
console.log('=== DIAGNOSTIC ===');
console.log('Chrome Extensions API available:', typeof chrome !== 'undefined' && typeof chrome.runtime !== 'undefined');
console.log('Total links on page:', document.querySelectorAll('a[href]').length);
console.log('Chrome version:', navigator.userAgent);
```

This will help identify if it's an API issue, link detection issue, or browser compatibility issue.
