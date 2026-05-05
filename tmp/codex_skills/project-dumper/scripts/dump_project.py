import os
import glob
import argparse
import datetime
from pathlib import Path

# 除外設定
IGNORED_DIRS = [
    ".git", ".venv", ".venv-gui", "node_modules", "__pycache__", 
    "build", "dist", ".pytest_cache", ".vscode", "temp", "tmp", 
    "logs", "assets", "image", ".serena"
]
IGNORED_FILES = ["package-lock.json", ".DS_Store"]
IGNORED_EXTS = [".png", ".ico", ".icns", ".pyc", ".db", ".sqlite", ".exe", ".bin", ".dll"]

def is_ignored(path):
    parts = path.split(os.sep)
    
    # ディレクトリ除外チェック
    for part in parts:
        if part in IGNORED_DIRS:
            return True
            
    # ファイル名チェック
    filename = os.path.basename(path)
    if filename in IGNORED_FILES:
        return True
        
    # 拡張子チェック
    _, ext = os.path.splitext(filename)
    if ext.lower() in IGNORED_EXTS:
        return True
        
    return False

def main():
    parser = argparse.ArgumentParser(description="Dump project files into a single Markdown file.")
    parser.add_argument("--output", "-o", help="Output file path. Defaults to ~/Downloads/project_dump_<timestamp>.md")
    parser.add_argument("--root", "-r", default=".", help="Root directory to scan. Defaults to current directory.")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    
    if args.output:
        output_file = os.path.abspath(os.path.expanduser(args.output))
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.abspath(os.path.expanduser(f"~/Downloads/project_dump_{timestamp}.md"))

    print(f"Scanning directory: {root_dir}")
    
    # ファイル収集
    all_files = glob.glob(os.path.join(root_dir, "**/*"), recursive=True)
    valid_files = [f for f in all_files if os.path.isfile(f) and not is_ignored(os.path.relpath(f, root_dir))]
    valid_files.sort()
    
    print(f"Found {len(valid_files)} target files.")
    
    try:
        # ディレクトリがない場合は作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write(f"# Project Dump\n")
            outfile.write(f"Source: {root_dir}\n")
            outfile.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for file_path in valid_files:
                rel_path = os.path.relpath(file_path, root_dir)
                try:
                    # バイナリ判定（簡易）
                    is_binary = False
                    with open(file_path, "rb") as f:
                        chunk = f.read(1024)
                        if b'\0' in chunk:
                            is_binary = True
                    
                    if is_binary:
                        print(f"Skipping binary file: {rel_path}")
                        continue

                    content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                    
                    outfile.write(f"\n# File: {rel_path}\n\n")
                    
                    # 拡張子に応じた言語指定
                    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
                    if not ext: ext = "txt"
                    
                    outfile.write(f"```{ext}\n")
                    outfile.write(content)
                    if not content.endswith("\n"):
                        outfile.write("\n")
                    outfile.write("```\n")
                    
                except Exception as e:
                    print(f"Error processing {rel_path}: {e}")
                    outfile.write(f"\n# File: {rel_path}\n\nError reading file: {e}\n")

        print(f"Successfully created: {output_file}")
        
    except IOError as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
