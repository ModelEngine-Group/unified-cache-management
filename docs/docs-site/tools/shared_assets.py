"""Keep shared media available without falling back to untranslated pages."""

from pathlib import Path

from mkdocs.plugins import event_priority
from mkdocs.structure.files import File


@event_priority(-200)  # Run after mkdocs-static-i18n filters each language tree.
def on_files(files, config):
    docs_dir = Path(config.docs_dir)
    for path in sorted((docs_dir / "assets").rglob("*")):
        if not path.is_file():
            continue
        uri = path.relative_to(docs_dir).as_posix()
        asset = File(uri, config.docs_dir, config.site_dir, config.use_directory_urls)
        if not asset.is_documentation_page() and files.get_file_from_path(uri) is None:
            files.append(asset)
    return files
