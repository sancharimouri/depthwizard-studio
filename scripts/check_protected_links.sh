#!/usr/bin/env bash
# The 5 PROTECTED LINKS (owner's standing rule, 2026-10-01; CLAUDE.md, docs/HANDOFF.md §0).
# Run before ANY deploy, push, release or merge. Exit 0 = all pass; 1 = stop and tell the owner.
#
#   scripts/check_protected_links.sh            # live URLs + the local web build (builds it if missing)
#   SKIP_BUILD_CHECK=1 scripts/check_protected_links.sh   # live URLs only
set -u
cd "$(dirname "$0")/.."

URLS=(
  "https://github.com/sancharimouri/depthwizard2-desktop/releases/latest"
  "https://depthwizard-studio.vercel.app/"
  "https://depthwizard-studio.vercel.app/#demo-video"
  "https://github.com/sancharimouri/depthwizard-studio"
  "https://depthwizard-studio.vercel.app/#/demo"
)
fail=0
say() { printf '%-6s %s\n' "$1" "$2"; }

echo "== live URLs (HTTP 200 after redirects)"
for u in "${URLS[@]}"; do
  # the #fragment never reaches the server: the page must load, and the build check below proves the anchor/route
  read -r code final < <(curl -sSL -o /dev/null --max-time 30 -A "dw2-protected-links-check" \
      -w '%{http_code} %{url_effective}' "$u" 2>/dev/null || echo "000 -")
  if [ "$code" = "200" ]; then say PASS "$code  $u"; else say FAIL "$code  $u"; fail=1; fi
  case "$u" in
    */releases/latest)  # must resolve to a concrete release tag, not the empty releases list
      if [[ "$final" == */releases/tag/* ]]; then say PASS "     -> $final"
      else say FAIL "     -> $final (no published latest release)"; fail=1; fi ;;
  esac
done

echo "== live page content"
if curl -sSL --max-time 30 "https://depthwizard-studio.vercel.app/" 2>/dev/null | grep -q 'id="demo-video"'; then
  say PASS 'live https://depthwizard-studio.vercel.app/ serves id="demo-video"'
else say FAIL 'live site HTML has no id="demo-video"'; fail=1; fi

if [ "${SKIP_BUILD_CHECK:-0}" != "1" ]; then
  echo "== local web build (frontend/dist)"
  if [ ! -f frontend/dist/index.html ]; then
    (cd frontend && npm run build:web >/dev/null 2>&1) || { say FAIL "npm run build:web failed"; fail=1; }
  fi
  if grep -q 'id="demo-video"' frontend/dist/index.html 2>/dev/null; then say PASS 'dist/index.html has id="demo-video"'
  else say FAIL 'dist/index.html is missing id="demo-video"'; fail=1; fi
  if grep -q '"page-explore":`#/demo`\|"page-explore":"#/demo"' frontend/dist/assets/*.js 2>/dev/null; then say PASS 'dist JS routes page-explore -> #/demo'
  else say FAIL 'dist JS has no #/demo route for page-explore'; fail=1; fi
  if grep -q '"page-explore": "#/demo"' frontend/src/routes.js && grep -q 'DEMO_VIDEO_ANCHOR = "demo-video"' frontend/src/routes.js; then
    say PASS 'src/routes.js keeps #/demo and the demo-video anchor'
  else say FAIL 'src/routes.js lost #/demo or the demo-video anchor'; fail=1; fi
fi

echo "== $( [ $fail = 0 ] && echo 'ALL PROTECTED LINKS OK' || echo 'PROTECTED LINK CHECK FAILED: stop and tell the owner' )"
exit $fail
