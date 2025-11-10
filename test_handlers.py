#!/usr/bin/env python3
"""
Test script for .txt and .mcfunction handlers without requiring translation model.
"""

import json
from pathlib import Path

# Test TextHandler line-by-line parsing
print("=" * 80)
print("TEST 1: TextHandler - Line-by-line parsing")
print("=" * 80)

test_txt_content = """# Test .txt file
# Comments should not be translated

Welcome to the server!
Please read the rules carefully.

Have fun and enjoy your stay.
"""

lines = test_txt_content.split('\n')
print(f"Total lines: {len(lines)}")

translatable_lines = []
for line_num, line in enumerate(lines, 1):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        print(f"Line {line_num}: SKIP (empty or comment)")
    else:
        print(f"Line {line_num}: TRANSLATE: {line}")
        translatable_lines.append(line)

print(f"\nResult: {len(translatable_lines)}/{len(lines)} lines translatable")

# Test McfunctionHandler JSON extraction
print("\n" + "=" * 80)
print("TEST 2: McfunctionHandler - JSON extraction")
print("=" * 80)

test_commands = [
    'tellraw @a {"text":"Welcome to the server!","color":"gold"}',
    'title @a title {"text":"Server Event","bold":true}',
    'tellraw @a [{"text":"Player ","color":"gray"},{"text":"has joined","color":"green"}]',
    'gamemode survival @a',  # No JSON
    'tellraw @a {"text":"Thank you!"}',
]

def extract_json_from_command(line: str):
    """Extract JSON components from command."""
    json_patterns = []
    stack = []
    start = None

    for i, char in enumerate(line):
        if char in ['{', '[']:
            if not stack:
                start = i
            stack.append(char)
        elif char in ['}', ']']:
            if stack:
                expected = '{' if char == '}' else '['
                if stack[-1] == expected:
                    stack.pop()
                    if not stack and start is not None:
                        json_str = line[start:i+1]
                        json_patterns.append(json_str)
                        start = None

    return json_patterns

for cmd_num, cmd in enumerate(test_commands, 1):
    print(f"\nCommand {cmd_num}: {cmd[:60]}...")
    json_components = extract_json_from_command(cmd)

    if not json_components:
        print("  → No JSON found")
    else:
        for json_str in json_components:
            print(f"  → JSON extracted: {json_str[:50]}...")
            try:
                data = json.loads(json_str)
                print(f"     Valid JSON: {data}")

                # Find 'text' fields
                def find_text_fields(obj, path=""):
                    texts = []
                    if isinstance(obj, dict):
                        if 'text' in obj:
                            texts.append((f"{path}.text", obj['text']))
                        for key, value in obj.items():
                            if isinstance(value, (dict, list)):
                                texts.extend(find_text_fields(value, f"{path}.{key}"))
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            if isinstance(item, (dict, list)):
                                texts.extend(find_text_fields(item, f"{path}[{i}]"))
                    return texts

                text_fields = find_text_fields(data)
                for path, text in text_fields:
                    print(f"     → Text field {path}: '{text}'")

            except json.JSONDecodeError as e:
                print(f"     ERROR: Invalid JSON - {e}")

# Test pattern filtering
print("\n" + "=" * 80)
print("TEST 3: Pattern filtering")
print("=" * 80)

from fnmatch import fnmatch

test_files = [
    "plugins/Essentials/config.yml",
    "plugins/Quests/lang/en_US.yml",
    "backup/old_config.yml",
    "datapacks/custom/data.json",
    "test.txt",
    "test.bak",
]

include_patterns = ["plugins/**", "datapacks/**"]
exclude_patterns = ["*.bak", "backup/**"]

print(f"Files: {len(test_files)}")
print(f"Include patterns: {include_patterns}")
print(f"Exclude patterns: {exclude_patterns}")
print()

for file_path in test_files:
    # Check include
    included = not include_patterns or any(fnmatch(file_path, p) for p in include_patterns)

    # Check exclude
    excluded = any(fnmatch(file_path, p) for p in exclude_patterns)

    status = "✅ INCLUDE" if (included and not excluded) else "❌ SKIP"
    reason = ""
    if not included:
        reason = " (not in include)"
    elif excluded:
        reason = " (in exclude)"

    print(f"{status:12} {file_path}{reason}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
