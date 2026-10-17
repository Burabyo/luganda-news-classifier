#!/usr/bin/env bash
# Build report/report.pdf from report/report.html with headless Chrome.
# Usage: bash scripts/build_report.sh "https://your-video-link"
set -euo pipefail
cd "$(dirname "$0")/../report"
if [ $# -ge 1 ]; then
  python3 - "$1" <<'PY'
import re, sys
p = "report.html"; s = open(p, encoding="utf-8").read()
url = sys.argv[1]
s = re.sub(r'<span id="video-link">.*?</span>', f'<span id="video-link"><a href="{url}">{url}</a></span>', s)
open(p, "w", encoding="utf-8").write(s)
PY
fi
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --allow-file-access-from-files --print-to-pdf=report.pdf "file://$PWD/report.html"
echo "report/report.pdf written"
