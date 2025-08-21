#!/bin/bash

total=0
while read -r file; do
  lines=$(wc -l < "$file")
  echo "==== FILE: $file  (Lines: $lines) ===="
  cat "$file"
  echo
  total=$((total + lines))
done < <(find . -maxdepth 1 \( -name nn -o -name .git \) -prune -o -type f \( -name "*.md" -o -name "*.cpp" -o -name "*.py" -o -name "*.h" -o -name "*.sh" \) -print)

echo "==== TOTAL LINES: $total ===="
