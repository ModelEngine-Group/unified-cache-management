"""Link Docker deployment to the existing, localized Model Tour."""

from mkdocs.plugins import event_priority
from mkdocs.structure.nav import Link


@event_priority(-200)
def on_nav(nav, config, files):
    deployment = next(
        page
        for page in nav.pages
        if page.file.src_uri.endswith("user-guide/frameworks/index.md")
    )
    model_tour = next(
        page
        for page in nav.pages
        if page.file.src_uri.endswith("user-guide/model-tour/index.md")
    )
    # Reusing the Page in nav would overwrite its parent and reuse its title.
    docker = Link("Docker", model_tour.url)
    docker.parent = deployment.parent
    deployment.parent.children.insert(1, docker)
    return nav
