total=0
while read -r file; do
  lines=$(grep -v "0x" "$file" | wc -l)  # Count lines excluding those containing 0x
  echo "==== FILE: $file  (Lines: $lines) ===="
  grep -v "0x" "$file"  # Skip lines containing 0x in the content
  echo
  total=$((total + lines))
done < <(find . -mindepth 1 -maxdepth 1 \( -name .git \) -prune -o -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -print)

echo "==== TOTAL LINES: $total ===="
