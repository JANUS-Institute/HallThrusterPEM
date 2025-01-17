"""Generate the code reference pages and navigation."""
from pathlib import Path
import sys

import mkdocs_gen_files

LOCAL_TEST = False  # False if this is being run by mkdocs automatically
nav = mkdocs_gen_files.Nav()
import_name = "hallmd"
nav_file = "SUMMARY.md"
index_file_name = "index.md"
root = Path(__file__).parent.parent
src = root / "src" / import_name
mod_symbol_nav = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'
pack_symbol_nav = '<code class="doc-symbol doc-symbol-nav doc-symbol-package"></code>'
pack_symbol_head = '<code class="doc-symbol doc-symbol-heading doc-symbol-package"></code>'

if not src.exists():
    sys.exit(0)

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src)
    doc_path = path.relative_to(src).with_suffix(".md")
    module_parts = tuple(module_path.parts)

    if module_parts[-1] in ['__main__.py', 'dataloader.py']:
        continue

    # Construct doc navigation path based on src directory layout
    nav_parts = []     # For mkdocs literate-nav
    python_parts = []  # Like scipy.stats.normal
    for part in module_parts:
        if part == "__init__.py":
            doc_path = doc_path.with_name(index_file_name)
            if len(module_parts) == 1:
                nav_parts.append(f"{pack_symbol_nav} {import_name}")    # Top-level package
        elif part.endswith('.py'):
            python_parts.append(Path(part).stem)
            nav_parts.append(f"{mod_symbol_nav} {Path(part).stem}")     # Modules
        else:
            python_parts.append(part)
            nav_parts.append(f"{pack_symbol_nav} {part}")               # Subpackages

    nav[tuple(nav_parts)] = doc_path.as_posix()
    full_doc_path = Path('reference', doc_path)
    python_name = ".".join(python_parts)
    python_name = f'{import_name}{"." if python_name else ""}{python_name}'

    # Create virtual docs/reference/*.md files that are loaded by mkdocs
    if LOCAL_TEST:
        full_doc_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(full_doc_path, 'w')
    else:
        fd = mkdocs_gen_files.open(full_doc_path, "w")

    if doc_path.name == index_file_name:
        # Change md header for packages
        fd.write(f'# {pack_symbol_head} `{python_name}`\n')
        fd.write(f'::: {python_name}\n')
        fd.write(f'    options:\n')
        fd.write(f'      show_root_heading: false\n')
        fd.write(f'      show_root_toc_entry: false\n')
        fd.write(f'      heading_level: 2\n')
    else:
        fd.write(f'---\ntitle: {python_parts[-1]}\n---\n\n::: {python_name}')
    fd.close()

    if not LOCAL_TEST:
        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the SUMMARY.md file for literate-nav
fd = open(f'reference/{nav_file}', 'w') if LOCAL_TEST else mkdocs_gen_files.open(f"reference/{nav_file}", "w")
fd.writelines(nav.build_literate_nav())
fd.close()
