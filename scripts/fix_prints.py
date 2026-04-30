"""Bulk replace print() with logger.info() — safe version.

Only adds logging import at the TOP of the file (before first class/function),
never inside function bodies.
"""
import os
import re

base = r"c:\Users\kaush\OneDrive\Documents\safetymind\cognicore-my-openenv\cognicore"
skip_files = {"cli.py", "__init__.py"}
skip_dirs = {"envs/data", "core"}

files_fixed = []

for root, dirs, files in os.walk(base):
    for f in files:
        if not f.endswith(".py"):
            continue
        if f in skip_files:
            continue

        rel = os.path.relpath(os.path.join(root, f), base).replace("\\", "/")

        skip = False
        for sd in skip_dirs:
            if rel.startswith(sd):
                skip = True
        if skip:
            continue

        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        # Check if file has print() in indented code
        has_prints = any(
            re.match(r"^\s+print\(", line) for line in lines
        )
        if not has_prints:
            continue

        content = "".join(lines)
        count_before = sum(1 for line in lines if re.match(r"^\s+print\(", line))

        has_logging_import = "import logging" in content
        has_logger_var = re.search(r'^logger\s*=\s*logging\.getLogger', content, re.MULTILINE)

        module_name = "cognicore." + rel.replace("/", ".").replace(".py", "")

        # Step 1: Find insertion point — after last top-level import, before first class/def
        insert_lines = []
        if not has_logging_import:
            insert_lines.append("import logging\n")
        if not has_logger_var:
            insert_lines.append(f'\nlogger = logging.getLogger("{module_name}")\n')

        if insert_lines:
            # Find last top-level import line
            last_import_idx = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    # Make sure it's top-level (no leading whitespace)
                    if not line[0].isspace():
                        last_import_idx = i

            if last_import_idx >= 0:
                for j, insert_line in enumerate(insert_lines):
                    lines.insert(last_import_idx + 1 + j, insert_line)

        # Step 2: Replace print() with logger.info() — only indented prints
        new_lines = []
        for line in lines:
            if re.match(r"^\s+print\(f[\"']", line):
                line = re.sub(r"^(\s+)print\(", r"\1logger.info(", line)
            elif re.match(r"^\s+print\([\"']", line):
                line = re.sub(r"^(\s+)print\(", r"\1logger.info(", line)
            new_lines.append(line)

        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(new_lines)

        files_fixed.append((rel, count_before))

print(f"Fixed {len(files_fixed)} files:")
for f, count in files_fixed:
    print(f"  {f}: {count} print() -> logger.info()")
