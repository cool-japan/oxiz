# Browser Testing Guide for OxiZ WASM

This guide provides instructions for testing OxiZ WASM across different browser environments.

## Quick Start

```bash
# Build for web target
./build.sh release web

# Start a local server
python3 -m http.server 8000

# Open http://localhost:8000/examples/playground.html
```

## Supported Browsers

### Minimum Requirements

| Browser | Minimum Version | Notes |
|---------|----------------|-------|
| Chrome | 67+ | Full support |
| Firefox | 68+ | Full support |
| Safari | 14+ | BigInt support required |
| Edge | 79+ | Chromium-based |
| Opera | 54+ | Full support |

### Feature Requirements

- **WebAssembly**: All modern browsers
- **ES Modules**: Chrome 61+, Firefox 60+, Safari 11+
- **BigInt**: Chrome 67+, Firefox 68+, Safari 14+
- **Web Workers**: All modern browsers

## Testing Environments

### 1. Chrome/Chromium Testing

#### Desktop Chrome
```bash
# Open with specific profile
google-chrome --user-data-dir=/tmp/chrome-test examples/playground.html

# With DevTools
google-chrome --auto-open-devtools-for-tabs examples/playground.html

# Disable cache
google-chrome --disable-cache examples/playground.html
```

#### Headless Chrome (for CI)
```bash
# Install puppeteer
npm install -D puppeteer

# Run test script
node test-chrome.js
```

**test-chrome.js:**
```javascript
const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    // Listen for console messages
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));

    // Navigate to test page
    await page.goto('http://localhost:8000/examples/playground.html');

    // Wait for WASM to load
    await page.waitForFunction(() => window.solver !== null, { timeout: 10000 });

    // Run tests
    const result = await page.evaluate(() => {
        return window.solver.execute('(check-sat)');
    });

    console.log('Test result:', result);

    await browser.close();
})();
```

### 2. Firefox Testing

#### Desktop Firefox
```bash
# Open with specific profile
firefox -P test-profile examples/playground.html

# Private browsing
firefox -private-window examples/playground.html

# Developer Edition
firefox-developer-edition examples/playground.html
```

#### Headless Firefox
```bash
firefox -headless -screenshot http://localhost:8000/examples/playground.html
```

#### Using Playwright
```javascript
const { firefox } = require('playwright');

(async () => {
    const browser = await firefox.launch();
    const page = await browser.newPage();
    await page.goto('http://localhost:8000/examples/playground.html');

    // Run tests...

    await browser.close();
})();
```

### 3. Safari Testing

#### Desktop Safari (macOS)
```bash
# Open Safari from command line
open -a Safari examples/playground.html

# Enable Developer Menu
# Safari > Preferences > Advanced > Show Develop menu in menu bar
```

#### Safari Technology Preview
```bash
# Download from: https://developer.apple.com/safari/technology-preview/
open -a "Safari Technology Preview" examples/playground.html
```

#### Safari on iOS (Simulator)
```bash
# Install Xcode and iOS Simulator
# Start simulator
open -a Simulator

# Open Safari in simulator and navigate to:
# http://YOUR_IP:8000/examples/playground.html
```

### 4. Edge Testing

#### Desktop Edge (Chromium)
```bash
# Windows
start msedge examples/playground.html

# macOS
open -a "Microsoft Edge" examples/playground.html

# Linux
microsoft-edge examples/playground.html
```

#### Using Playwright
```javascript
const { chromium } = require('playwright');

(async () => {
    const browser = await chromium.launch({ channel: 'msedge' });
    const page = await browser.newPage();
    await page.goto('http://localhost:8000/examples/playground.html');

    // Run tests...

    await browser.close();
})();
```

## Automated Testing

### Using wasm-pack

```bash
# Test in Chrome (headless)
wasm-pack test --headless --chrome

# Test in Firefox (headless)
wasm-pack test --headless --firefox

# Test in Safari (requires safaridriver)
wasm-pack test --headless --safari

# Test with specific features
wasm-pack test --headless --chrome --features "experimental"
```

### Using Playwright (Recommended)

**playwright.config.js:**
```javascript
module.exports = {
    projects: [
        {
            name: 'chromium',
            use: { browserName: 'chromium' },
        },
        {
            name: 'firefox',
            use: { browserName: 'firefox' },
        },
        {
            name: 'webkit',
            use: { browserName: 'webkit' },
        },
    ],
};
```

**tests/browser.spec.js:**
```javascript
const { test, expect } = require('@playwright/test');

test.describe('OxiZ WASM', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('http://localhost:8000/examples/playground.html');
        await page.waitForFunction(() => window.solver !== null);
    });

    test('should solve SAT problem', async ({ page }) => {
        const result = await page.evaluate(() => {
            return window.solver.execute('(check-sat)');
        });
        expect(result).toContain('sat');
    });

    test('should detect UNSAT', async ({ page }) => {
        const result = await page.evaluate(() => {
            return window.solver.execute(`
                (declare-const x Int)
                (assert (> x 5))
                (assert (< x 3))
                (check-sat)
            `);
        });
        expect(result).toContain('unsat');
    });
});
```

Run tests:
```bash
npm install -D @playwright/test
npx playwright install
npx playwright test
```

### Using Selenium

**selenium-test.js:**
```javascript
const { Builder, By, until } = require('selenium-webdriver');

async function testBrowser(browserName) {
    const driver = await new Builder().forBrowser(browserName).build();

    try {
        await driver.get('http://localhost:8000/examples/playground.html');

        // Wait for solver to be ready
        await driver.wait(
            until.elementLocated(By.css('.status.ready')),
            10000
        );

        // Click solve button
        await driver.findElement(By.id('solveBtn')).click();

        // Check output
        const output = await driver.findElement(By.id('output')).getText();
        console.log(`${browserName} result:`, output);
    } finally {
        await driver.quit();
    }
}

// Test all browsers
(async () => {
    for (const browser of ['chrome', 'firefox', 'safari', 'MicrosoftEdge']) {
        try {
            await testBrowser(browser);
        } catch (error) {
            console.error(`${browser} failed:`, error.message);
        }
    }
})();
```

## Manual Testing Checklist

### Basic Functionality
- [ ] WASM module loads successfully
- [ ] Solver initializes without errors
- [ ] Simple SAT problem solves correctly
- [ ] UNSAT problem detected correctly
- [ ] Model extraction works
- [ ] Error messages are clear

### Performance
- [ ] Initial load time < 3 seconds
- [ ] Simple problems solve in < 100ms
- [ ] Complex problems complete without hanging
- [ ] No memory leaks after multiple solves

### UI/UX
- [ ] Examples load correctly
- [ ] Buttons respond to clicks
- [ ] Output displays properly
- [ ] Async operations don't block UI
- [ ] Mobile responsive design works

### Advanced Features
- [ ] Incremental solving (push/pop)
- [ ] Optimization problems
- [ ] Unsat core extraction
- [ ] Proof generation
- [ ] Statistics collection

## Testing Tools

### Browser DevTools

#### Chrome DevTools
```javascript
// Check WASM memory
performance.memory.usedJSHeapSize

// Profile solve time
console.time('solve');
solver.execute('(check-sat)');
console.timeEnd('solve');

// Check for memory leaks
// Memory tab > Take heap snapshot > Compare snapshots
```

#### Firefox DevTools
```javascript
// Performance profiling
// DevTools > Performance > Start Recording > Solve > Stop

// WASM debugging
// DevTools > Debugger > wasm://
```

### BrowserStack

For testing on real devices and older browsers:

1. Sign up at https://www.browserstack.com
2. Use their Live or Automate features
3. Test on various OS/browser combinations

### Sauce Labs

Similar to BrowserStack:

1. Sign up at https://saucelabs.com
2. Run automated tests across browsers
3. View test results and recordings

## Common Issues

### Issue: WASM fails to load in Safari

**Solution**: Ensure you're serving with correct MIME types:

```nginx
# nginx config
types {
    application/wasm wasm;
}
```

```python
# Python simple server with correct MIME type
import http.server
import socketserver

class WasmHandler(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        mimetype = super().guess_type(path)
        if path.endswith('.wasm'):
            return 'application/wasm'
        return mimetype

with socketserver.TCPServer(("", 8000), WasmHandler) as httpd:
    httpd.serve_forever()
```

### Issue: CORS errors

**Solution**: Use a proper HTTP server, not `file://` protocol

### Issue: BigInt errors in Safari < 14

**Solution**: Add BigInt polyfill or show browser upgrade message

```javascript
if (typeof BigInt === 'undefined') {
    alert('Your browser does not support BigInt. Please upgrade to a newer version.');
}
```

### Issue: Module loading fails

**Solution**: Ensure proper ES module support

```html
<!-- Use type="module" -->
<script type="module" src="./app.js"></script>

<!-- Or use import maps for older browsers -->
<script src="https://unpkg.com/es-module-shims@1.5.4/dist/es-module-shims.js"></script>
```

## CI/CD Integration

### GitHub Actions

**.github/workflows/browser-tests.yml:**
```yaml
name: Browser Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: wasm32-unknown-unknown

      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

      - name: Build WASM
        run: wasm-pack build --target web --release
        working-directory: ./oxiz-wasm

      - name: Install Playwright
        run: |
          npm install -D @playwright/test
          npx playwright install --with-deps

      - name: Start HTTP server
        run: |
          python3 -m http.server 8000 &
          sleep 2

      - name: Run tests
        run: npx playwright test

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-results
          path: test-results/
```

## Performance Benchmarking

### Lighthouse

```bash
# Install lighthouse
npm install -g lighthouse

# Run audit
lighthouse http://localhost:8000/examples/playground.html \
    --output html \
    --output-path ./lighthouse-report.html
```

### WebPageTest

1. Visit https://www.webpagetest.org
2. Enter your URL
3. Select browsers and locations to test
4. Analyze results

## Debugging Tips

1. **Use verbose logging**:
   ```javascript
   solver.setTracing(true);
   ```

2. **Check WASM size**:
   ```bash
   ls -lh pkg/*.wasm
   wasm-opt --print-size pkg/oxiz_wasm_bg.wasm
   ```

3. **Profile memory**:
   ```javascript
   console.profile('solve');
   solver.execute(script);
   console.profileEnd('solve');
   ```

4. **Test in incognito/private mode** to avoid extension interference

5. **Disable cache** during development

## Resources

- [MDN Browser Compatibility](https://developer.mozilla.org/en-US/docs/WebAssembly#browser_compatibility)
- [Can I Use WebAssembly](https://caniuse.com/wasm)
- [wasm-pack Documentation](https://rustwasm.github.io/wasm-pack/)
- [Playwright Documentation](https://playwright.dev)
