"""
Post-build optimization for ICP WASM:
Strip unused RTTI global exports and remove unreferenced globals via WAT round-trip.

This is a safe alternative to wasm-opt which corrupts the code. We:
1. Strip non-exception RTTI global exports (typeinfo, vtable)
2. Remove unreferenced global definitions and re-index

No wasm-opt is used — only wasm-tools print/parse for lossless WAT round-trip.
"""

import re
import subprocess
import shutil
from pathlib import Path
from icpp import icpp_toml


def count_globals(wasm_path: Path) -> int:
    """Count defined globals in WASM using wasm-tools (matches ICP counting)"""
    try:
        result = subprocess.run(
            ["wasm-tools", "print", str(wasm_path)],
            capture_output=True, text=True, timeout=120
        )
        return len(re.findall(r'\(global \(;\d+;\)', result.stdout))
    except Exception:
        return -1


def is_exception_rtti(export_name: str) -> bool:
    """Check if an RTTI export is needed for C++ exception handling."""
    return ('exception' in export_name or
            'error' in export_name or
            'bad_' in export_name or
            '__cxxabiv1' in export_name or
            'nested_exception' in export_name)


def optimize_globals(wasm_path: Path, output_path: Path) -> tuple:
    """Strip unused RTTI globals and re-index. Returns (removed_exports, removed_globals)."""
    result = subprocess.run(
        ["wasm-tools", "print", str(wasm_path)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"wasm-tools print failed: {result.stderr}")

    wat = result.stdout
    lines = wat.split('\n')

    # Step 1: Find which globals are referenced by code (global.get/set)
    used_globals = set()
    for m in re.finditer(r'global\.(get|set) (\d+)', wat):
        used_globals.add(int(m.group(2)))

    # Step 2: Find which exports to keep/remove
    exported_globals = set()
    export_lines_to_remove = set()
    removed_exports = 0
    for i, line in enumerate(lines):
        if '(export' in line and '(global' in line:
            m_idx = re.search(r'\(global (\d+)\)', line)
            m_name = re.search(r'"([^"]+)"', line)
            if m_idx and m_name:
                idx = int(m_idx.group(1))
                name = m_name.group(1)
                if ('_ZTV' in name or '_ZTI' in name or '_ZTS' in name):
                    if not is_exception_rtti(name):
                        export_lines_to_remove.add(i)
                        removed_exports += 1
                        continue
                exported_globals.add(idx)

    # Step 3: Find all defined globals
    all_globals = {}  # idx -> line_number
    for i, line in enumerate(lines):
        m = re.match(r'\s+\(global \(;(\d+);\)', line)
        if m:
            all_globals[int(m.group(1))] = i

    # Step 4: Determine removable globals
    needed = used_globals | exported_globals
    removable = set(all_globals.keys()) - needed
    removable_lines = {all_globals[idx] for idx in removable}

    # Step 5: Build old->new index mapping
    old_to_new = {}
    new_idx = 0
    for old_idx in sorted(all_globals.keys()):
        if old_idx not in removable:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    # Step 6: Rewrite WAT
    new_lines = []
    for i, line in enumerate(lines):
        if i in removable_lines or i in export_lines_to_remove:
            continue

        # Renumber global references
        line = re.sub(
            r'global\.(get|set) (\d+)',
            lambda m: f"global.{m.group(1)} {old_to_new.get(int(m.group(2)), int(m.group(2)))}",
            line
        )
        line = re.sub(
            r'\(global (\d+)\)',
            lambda m: f"(global {old_to_new.get(int(m.group(1)), int(m.group(1)))})",
            line
        )
        line = re.sub(
            r'\(global \(;(\d+);\)',
            lambda m: f"(global (;{old_to_new[int(m.group(1))]};)" if int(m.group(1)) in old_to_new else m.group(0),
            line
        )

        new_lines.append(line)

    # Step 7: Assemble back to WASM
    wat_path = wasm_path.with_suffix(".wat")
    wat_path.write_text('\n'.join(new_lines))

    tmp_path = output_path.with_name(output_path.stem + "_tmp" + output_path.suffix)
    result = subprocess.run(
        ["wasm-tools", "parse", str(wat_path), "-o", str(tmp_path)],
        capture_output=True, text=True, timeout=120
    )
    wat_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"wasm-tools parse failed: {result.stderr}")

    # Step 8: Strip all custom sections to fix section ordering for llvm-objcopy
    # The WAT roundtrip can reorder custom sections, which confuses llvm-objcopy.
    # icpp will re-add the necessary custom sections after this function returns.
    result = subprocess.run(
        ["wasm-tools", "strip", "--all", str(tmp_path), "-o", str(output_path)],
        capture_output=True, text=True, timeout=60
    )
    tmp_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"wasm-tools strip failed: {result.stderr}")

    return removed_exports, len(removable)


def main() -> None:
    """Strip RTTI exports + remove unreferenced globals to fit ICP limit (1000)"""
    build_path = icpp_toml.icpp_toml_path.parent / "build"
    wasm_path = (build_path / f"{icpp_toml.build_wasm['canister']}.wasm").resolve()

    if not wasm_path.exists():
        print(f"WASM file not found: {wasm_path}")
        return

    size_before = wasm_path.stat().st_size
    globals_before = count_globals(wasm_path)
    print(f"Before: {size_before:,} bytes, {globals_before} globals")

    if globals_before <= 1000:
        print("Already within ICP limit — skipping optimization")
        return

    # Save backup
    backup_path = wasm_path.with_name(wasm_path.stem + "_before_opt" + wasm_path.suffix)
    shutil.copy(wasm_path, backup_path)

    try:
        optimized_path = wasm_path.with_name(wasm_path.stem + "_opt" + wasm_path.suffix)
        removed_exports, removed_globals = optimize_globals(wasm_path, optimized_path)
        shutil.move(str(optimized_path), str(wasm_path))

        size_after = wasm_path.stat().st_size
        globals_after = count_globals(wasm_path)

        print(f"Stripped {removed_exports} RTTI exports, removed {removed_globals} globals")
        print(f"After:  {size_after:,} bytes, {globals_after} globals")
        print(f"Delta:  {size_after - size_before:+,} bytes, {globals_after - globals_before:+} globals")

        if globals_after > 1000:
            print(f"\nWARNING: Still {globals_after} globals (ICP limit = 1000)")
        else:
            print(f"\nWithin ICP limit ({globals_after} <= 1000)")

    except Exception as e:
        print(f"Optimization error: {e}")
        if backup_path.exists():
            shutil.copy(backup_path, wasm_path)
            print("Restored original WASM from backup")


if __name__ == "__main__":
    main()
