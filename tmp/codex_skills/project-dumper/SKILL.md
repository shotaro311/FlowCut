---
name: project-dumper
description: Dumps all text files in the target directory into a single Markdown file. Useful for creating documentation or sharing codebase context.
---

# Project Dumper Skill

This skill scans a project directory, filters out common ignored files (like `node_modules`, `.git`, binary files), and consolidates the content of all text files into a single Markdown document.

## When to use

Use this skill when the user asks to:
- "Document all files in this directory."
- "Dump the project codebase."
- "Combine all files into one markdown for review."
- "Create a snapshot of the current code."

## Usage

Execute the included python script. You can optionally specify the output path.

```bash
# Default output to ~/Downloads/project_dump_<timestamp>.md
python scripts/dump_project.py

# Specify output file
python scripts/dump_project.py --output ~/Downloads/my_project_docs.md

# Specify root directory to scan (if different from current)
python scripts/dump_project.py --root /path/to/project
```
