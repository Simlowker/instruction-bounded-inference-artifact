#!/usr/bin/env python3
"""Strip unused globals from WASM by operating on WAT text format.

Removes global definitions indexed 145+ (which are not referenced in code)
and their corresponding export entries. Does NOT modify code, data, or
any other sections, so it preserves canister correctness.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    wasm_in = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/llama_cpp.wasm")
    wasm_out = Path(sys.argv[2]) if len(sys.argv) > 2 else wasm_in

    # Step 1: Convert to WAT
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False) as f:
        wat_path = Path(f.name)
    subprocess.check_call(["wasm-tools", "print", str(wasm_in), "-o", str(wat_path)])

    # Step 2: Read WAT and filter
    with open(wat_path) as f:
        lines = f.readlines()

    # Remove global definitions (;N;) where N >= 145
    # Format: "  (global (;N;) ..."
    global_pattern = re.compile(r"^\s+\(global \(;(\d+);\)")
    # Remove exports of globals N >= 145
    # Format: '  (export "..." (global N))'
    export_global_pattern = re.compile(r'^\s+\(export ".*" \(global (\d+)\)\)')

    filtered = []
    removed_globals = 0
    removed_exports = 0
    for line in lines:
        m = global_pattern.match(line)
        if m and int(m.group(1)) >= 145:
            removed_globals += 1
            continue
        m = export_global_pattern.match(line)
        if m and int(m.group(1)) >= 145:
            removed_exports += 1
            continue
        filtered.append(line)

    print(f"Removed {removed_globals} global definitions (indices 145+)")
    print(f"Removed {removed_exports} global exports (indices 145+)")

    # Step 3: Write filtered WAT
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        wat_filtered = Path(f.name)
        f.writelines(filtered)

    # Step 4: Convert back to WASM
    subprocess.check_call(["wasm-tools", "parse", str(wat_filtered), "-o", str(wasm_out)])

    # Cleanup
    wat_path.unlink()
    wat_filtered.unlink()

    # Verify
    result = subprocess.run(
        ["wasm-tools", "print", str(wasm_out)],
        capture_output=True, text=True
    )
    global_count = len(re.findall(r"^  \(global ", result.stdout, re.MULTILINE))
    print(f"Final global count: {global_count}")
    print(f"Output: {wasm_out} ({wasm_out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
