# OxiZ WASM Release Checklist

This document provides a step-by-step checklist for releasing oxiz-wasm to NPM and deploying the playground.

## Pre-Release Checklist

### Code Quality
- [x] All tests passing (`cargo test`)
- [x] No compiler warnings in oxiz-wasm
- [x] No clippy warnings (`cargo clippy`)
- [x] Code formatted (`cargo fmt`)
- [x] Documentation complete
- [x] Examples working

### Version Management
- [ ] Update version in `Cargo.toml`
- [ ] Update version in `package.json`
- [ ] Update `CHANGELOG.md` (if exists)
- [ ] Update `Last Updated` date in `TODO.md`

Use the automated script:
```bash
./version-bump.sh patch  # or minor, or major
```

### Build Verification
- [ ] Clean build succeeds
  ```bash
  cargo clean
  cargo build --release
  ```

- [ ] WASM build succeeds for all targets
  ```bash
  ./build.sh all
  ```

- [ ] Optimized build completes
  ```bash
  ./build.sh optimized web
  ```

- [ ] Final WASM size is acceptable (target <2MB)
  ```bash
  ls -lh pkg-web/oxiz_wasm_bg.wasm
  ```

### Testing
- [ ] Unit tests pass
  ```bash
  cargo test
  ```

- [ ] Browser tests pass (Chrome)
  ```bash
  wasm-pack test --headless --chrome
  ```

- [ ] Browser tests pass (Firefox)
  ```bash
  wasm-pack test --headless --firefox
  ```

- [ ] Manual testing in playground
  - Open `examples/playground.html`
  - Test all example problems
  - Verify no console errors
  - Check performance is acceptable

- [ ] Test in multiple browsers
  - Chrome
  - Firefox
  - Safari (macOS/iOS)
  - Edge

See [BROWSER_TESTING.md](./BROWSER_TESTING.md) for detailed testing guide.

### Documentation
- [ ] README.md is up to date
- [ ] API documentation is complete
- [ ] Examples are working
- [ ] Tutorial links are valid
- [ ] TypeScript declarations are accurate

## NPM Publishing

### Setup
1. [ ] Ensure you have an NPM account
   ```bash
   npm login
   ```

2. [ ] Verify package name is available (first release only)
   ```bash
   npm view oxiz-wasm
   ```

### Dry Run
3. [ ] Run publish dry-run
   ```bash
   npm publish --dry-run
   ```

4. [ ] Verify package contents
   - Check `files` in package.json
   - Ensure no unnecessary files are included
   - Verify WASM files are present

### Publish
5. [ ] Run the automated publish script
   ```bash
   ./publish.sh
   ```

   This script will:
   - Check NPM authentication
   - Verify version is not already published
   - Run tests
   - Build all targets
   - Optimize builds
   - Show package sizes
   - Run dry-run
   - Confirm before publishing
   - Create git tag
   - Publish to NPM

6. [ ] Verify publication
   ```bash
   npm view oxiz-wasm
   ```

7. [ ] Push git tag
   ```bash
   git push origin vX.Y.Z
   ```

## Post-Publication

### CDN Verification
- [ ] Wait 5-10 minutes for CDN propagation

- [ ] Verify unpkg
  ```
  https://unpkg.com/oxiz-wasm@X.Y.Z/
  ```

- [ ] Verify jsDelivr
  ```
  https://cdn.jsdelivr.net/npm/oxiz-wasm@X.Y.Z/
  ```

- [ ] Test CDN loading
  Create a simple HTML file:
  ```html
  <script type="module">
    import init from 'https://unpkg.com/oxiz-wasm@X.Y.Z/pkg/oxiz_wasm.js';
    await init();
    console.log('Loaded from CDN!');
  </script>
  ```

### GitHub Release
- [ ] Create GitHub release
  ```bash
  gh release create vX.Y.Z \
    --title "Release vX.Y.Z" \
    --notes "Release notes here"
  ```

- [ ] Attach artifacts (optional)
  - Optimized WASM binary
  - TypeScript declarations
  - Examples zip

### Documentation Updates
- [ ] Update main README with new version
- [ ] Update getting started guide
- [ ] Update CDN examples with new version
- [ ] Verify all documentation links work

### Announcement
- [ ] Announce on project discussion board
- [ ] Update project website (if any)
- [ ] Post on social media (if applicable)
- [ ] Update crates.io listing (if published)

## Playground Deployment

### Option 1: GitHub Pages
1. [ ] Create `gh-pages` branch
   ```bash
   git checkout --orphan gh-pages
   ```

2. [ ] Copy playground and build artifacts
   ```bash
   cp -r examples/playground.html index.html
   cp -r pkg ./
   ```

3. [ ] Commit and push
   ```bash
   git add .
   git commit -m "Deploy playground"
   git push origin gh-pages
   ```

4. [ ] Enable GitHub Pages in repository settings
   - Settings > Pages
   - Source: gh-pages branch
   - Save

5. [ ] Verify deployment
   ```
   https://your-org.github.io/oxiz/
   ```

### Option 2: Netlify
1. [ ] Create `netlify.toml`
   ```toml
   [build]
     publish = "dist"
     command = "./build.sh optimized web && mkdir -p dist && cp examples/playground.html dist/index.html && cp -r pkg dist/"

   [[redirects]]
     from = "/*"
     to = "/index.html"
     status = 200
   ```

2. [ ] Deploy
   ```bash
   netlify deploy --prod
   ```

### Option 3: Vercel
1. [ ] Install Vercel CLI
   ```bash
   npm i -g vercel
   ```

2. [ ] Deploy
   ```bash
   vercel --prod
   ```

3. [ ] Configure
   - Build Command: `./build.sh optimized web`
   - Output Directory: `pkg`

## Rollback Procedure

If issues are found after release:

### Option 1: Deprecate Version
```bash
npm deprecate oxiz-wasm@X.Y.Z "This version has issues, use X.Y.Z+1 instead"
```

### Option 2: Unpublish (within 72 hours)
```bash
npm unpublish oxiz-wasm@X.Y.Z
```

**Note**: Only use this for critical security issues.

### Option 3: Publish Patch
1. Fix the issue
2. Bump to X.Y.Z+1
3. Publish new version
4. Deprecate old version

## Monitoring

### Package Health
- [ ] Check NPM download stats
  ```
  https://npm-stat.com/charts.html?package=oxiz-wasm
  ```

- [ ] Monitor GitHub issues for bug reports

- [ ] Check CDN hit rates
  ```
  https://www.jsdelivr.com/package/npm/oxiz-wasm
  ```

### Performance
- [ ] Monitor bundle size over time
- [ ] Check load times in different regions
- [ ] Verify WASM execution performance

## Next Release Planning

After a successful release:

1. [ ] Create milestone for next version
2. [ ] Triage open issues
3. [ ] Plan new features
4. [ ] Update TODO.md with next priorities
5. [ ] Set up beta/alpha releases if needed

## Emergency Contacts

- NPM Support: support@npmjs.com
- GitHub Support: https://support.github.com
- CDN Issues: Check respective CDN status pages

## Notes

- Always test on a staging environment first
- Keep release notes detailed and user-friendly
- Maintain semantic versioning (MAJOR.MINOR.PATCH)
- Coordinate releases with dependent packages
- Announce breaking changes well in advance

## Automation Ideas

Future improvements to automate:

- [ ] Automated version bumping in CI
- [ ] Automated changelog generation
- [ ] Automated browser testing in CI
- [ ] Automated CDN verification
- [ ] Automated GitHub release creation
- [ ] Automated documentation deployment
- [ ] Automated performance regression testing
