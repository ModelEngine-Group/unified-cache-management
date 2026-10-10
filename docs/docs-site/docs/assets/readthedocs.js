// Use RTD's published versions and translations in Material's header menus.
(function () {
  function renderNavigation(data) {
    const header = document.querySelector(".md-header__inner");
    const template = document.getElementById("ucm-rtd-navigation");
    if (!header || !template) return;

    // RTD's resolver preserves the current page when switching language/version.
    const filename = (
      data.readthedocs.resolver.filename ||
      document.querySelector('meta[name="readthedocs-resolver-filename"]')?.content ||
      ""
    ).replace(/\/index\.html$/, "/").replace(/^\//, "");
    const pageUrl = (root) => new URL(filename, root).href;
    const menus = template.content.cloneNode(true);
    const versionMenu = menus.querySelector(".ucm-rtd-version");
    versionMenu.querySelector("button").textContent = data.versions.current.slug;
    const versions = [
      data.versions.current,
      ...data.versions.active.filter(
        (version) => version.built && version.slug !== data.versions.current.slug
      ),
    ];
    for (const version of versions) {
      const item = document.createElement("li");
      item.className = "md-version__item";
      const link = document.createElement("a");
      link.className = "md-version__link";
      link.textContent = version.slug;
      link.href = pageUrl(version.urls.documentation);
      if (version.slug === data.versions.current.slug) {
        link.setAttribute("aria-current", "true");
      }
      item.append(link);
      versionMenu.querySelector("ul").append(item);
    }

    const languageMenu = menus.querySelector(".ucm-rtd-language");
    const projects = [data.projects.current, ...data.projects.translations].sort(
      (a, b) => a.language.code.localeCompare(b.language.code)
    );
    for (const project of projects) {
      const item = document.createElement("li");
      item.className = "md-select__item";
      const link = document.createElement("a");
      link.className = "md-select__link";
      link.textContent = project.language.code === "zh-cn"
        ? "简体中文" : project.language.name;
      link.hreflang = project.language.code;
      link.href = pageUrl(project.urls.documentation);
      if (project.slug === data.projects.current.slug) {
        link.setAttribute("aria-current", "true");
      }
      item.append(link);
      languageMenu.querySelector("ul").append(item);
    }

    header.querySelector(".ucm-rtd-version")?.remove();
    header.querySelector(".ucm-rtd-language")?.remove();
    header.querySelector(".md-header__title").after(versionMenu);
    if (projects.length > 1) {
      const search = header.querySelector('label[for="__search"]');
      (search || header.querySelector(".md-header__source")).before(languageMenu);
    }
    // Hide only the default flyout, after its replacement is ready.
    document.documentElement.dataset.rtdNavigation = "header";

    const input = document.querySelector(".md-search__input");
    if (input && !input.dataset.rtdSearch) {
      input.dataset.rtdSearch = "true";
      input.addEventListener("focus", function () {
        document.dispatchEvent(new CustomEvent("readthedocs-search-show"));
      });
    }
  }

  document.addEventListener("readthedocs-addons-data-ready", function (event) {
    renderNavigation(event.detail.data());
  });
  if (window.ReadTheDocsEventData) {
    renderNavigation(window.ReadTheDocsEventData.data());
  }
})();
