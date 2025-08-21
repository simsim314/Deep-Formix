#!/bin/bash

# Extensions whose content we will print
CONTENT_EXTS=(-iname "*.py" -o -iname "*.cpp" -o -iname "*.cc" -o -iname "*.c" -o -iname "*.hpp" -o -iname "*.h" -o -iname "*.sh" -o -iname "*.md")

total=0

echo "==== DIRECTORIES (depth 1) ===="
find . -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort

echo
echo "==== FILES (names only, non-code, depth 1) ===="
find . -maxdepth 1 -type f ! \( "${CONTENT_EXTS[@]}" \) -printf "%f\n" | sort

echo
echo "==== CODE/TEXT FILES (with content, depth 1) ===="
# Use process substitution so the while runs in the current shell (total is preserved)
while IFS= read -r -d '' file; do
  lines=$(wc -l < "$file")
  echo "==== FILE: ${file#./}  (Lines: $lines) ===="
  cat -- "$file"
  echo
  total=$(( total + lines ))
done < <(find . -maxdepth 1 -type f \( "${CONTENT_EXTS[@]}" \) -print0 | sort -z)

echo "==== TOTAL LINES (code/text files): $total ===="
