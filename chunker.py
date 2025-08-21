#!/usr/bin/env python3
import sys
import argparse
import subprocess
import platform
from typing import List

import pyperclip
from pynput import keyboard

BANNER = "BE IDLE wait for next chunk of data do not respond in length !"

BASH_CMD = r'''
total=0
while read -r file; do
  lines=$(grep -v "0x" "$file" | wc -l)
  echo "==== FILE: $file  (Lines: $lines) ===="
  grep -v "0x" "$file"
  echo
  total=$((total + lines))
done < <(find . \( -name .git \) -prune -o -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" -o -name "*.h" -o -name "*.cpp" \) -print)

echo "==== TOTAL LINES: $total ===="
'''.strip()


def run_bash_and_capture() -> List[str]:
    proc = subprocess.run(
        ["bash", "-lc", BASH_CMD],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.stdout.splitlines()


def chunk_lines(lines: List[str], n: int) -> List[str]:
    chunks = []
    total = (len(lines) + n - 1) // n if lines else 0
    for i in range(total):
        body = "\n".join(lines[i * n : (i + 1) * n])
        header = f"{BANNER}\n{i+1}/{total}"
        chunks.append(f"{header}\n{body}" if body else header)
    return chunks


class SpacePaster:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.idx = -1  # start before first
        self.is_macos = platform.system() == "Darwin"
        self.space_down = False
        self.kb = keyboard.Controller()
        self.listener = None  # store listener reference

    def on_press(self, key):
        if key == keyboard.Key.space and not self.space_down:
            self.space_down = True
            self._advance_and_paste()
        return True

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.space_down = False
        return True

    def _advance_and_paste(self):
        next_idx = self.idx + 1
        if next_idx >= len(self.chunks):
            print("All chunks pasted. Exiting.")
            if self.listener:
                self.listener.stop()
            return
        self.idx = next_idx
        pyperclip.copy(self.chunks[self.idx])
        print(f"Pasted chunk {self.idx+1}/{len(self.chunks)}")
        self._send_paste()

    def _send_paste(self):
        if self.is_macos:
            with self.kb.pressed(keyboard.Key.cmd):
                self.kb.press('v')
                self.kb.release('v')
        else:
            with self.kb.pressed(keyboard.Key.ctrl):
                self.kb.press('v')
                self.kb.release('v')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--chunk-lines", type=int, default=2400)
    args = ap.parse_args()

    lines = run_bash_and_capture()
    chunks = chunk_lines(lines, args.chunk_lines)
    if not chunks:
        print("No data; nothing to paste.")
        return

    sp = SpacePaster(chunks)

    print(f"Ready. {len(chunks)} chunk(s) prepared.")
    print("Press SPACE to copy+paste the NEXT chunk. Ctrl+C to exit early.")

    with keyboard.Listener(
        on_press=sp.on_press,
        on_release=sp.on_release,
        suppress=True
    ) as listener:
        sp.listener = listener
        listener.join()


if __name__ == "__main__":
    main()
