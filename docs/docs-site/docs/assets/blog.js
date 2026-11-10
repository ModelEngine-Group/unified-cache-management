(function () {
  const list = document.querySelector(".ucm-blog");

  if (list) {
    const searchForm = document.querySelector("[data-blog-search]");

    if (searchForm) {
      const input = searchForm.querySelector("input[type='search']");
      const banners = Array.from(list.querySelectorAll(".ucm-blog-banner"));
      const months = Array.from(list.querySelectorAll(".ucm-blog-month"));
      const years = Array.from(list.querySelectorAll(".ucm-blog-year-block"));
      const titleText = new Map();

      banners.forEach(function (banner) {
        const title = banner.querySelector(".ucm-blog-banner__title");
        const text = title ? title.textContent : "";
        titleText.set(banner, text.replace(/\s+/g, " ").trim().toLowerCase());
      });

      const empty = document.createElement("p");
      empty.className = "ucm-blog-empty ucm-blog-search-empty";
      empty.hidden = true;
      empty.setAttribute("role", "status");
      empty.setAttribute("aria-live", "polite");
      empty.setAttribute("aria-atomic", "true");
      empty.textContent =
        searchForm.dataset.empty || "No posts match your search.";
      list.appendChild(empty);

      function clearFilter() {
        banners.forEach(function (banner) {
          banner.hidden = false;
        });
        months.forEach(function (month) {
          month.hidden = false;
        });
        years.forEach(function (year) {
          year.hidden = false;
        });
        empty.hidden = true;
      }

      function applySearch() {
        const term = input.value.replace(/\s+/g, " ").trim().toLowerCase();

        if (term === "") {
          clearFilter();
          return;
        }

        let matches = 0;

        months.forEach(function (month) {
          let monthMatches = 0;

          month.querySelectorAll(".ucm-blog-banner").forEach(function (banner) {
            const hit = (titleText.get(banner) || "").includes(term);
            banner.hidden = !hit;
            if (hit) {
              monthMatches += 1;
            }
          });

          month.hidden = monthMatches === 0;
          matches += monthMatches;
        });

        years.forEach(function (year) {
          year.hidden = year.querySelectorAll(".ucm-blog-month:not([hidden])").length === 0;
        });

        empty.hidden = matches > 0;
      }

      searchForm.addEventListener("submit", function (event) {
        event.preventDefault();
        applySearch();
      });

      input.addEventListener("input", function (event) {
        if (!event.isComposing) {
          applySearch();
        }
      });

      input.addEventListener("compositionend", function () {
        window.setTimeout(applySearch, 0);
      });

      input.addEventListener("search", applySearch);
      input.addEventListener("change", applySearch);
    }
  }

  const timeline = document.querySelector(".ucm-blog-timeline");

  if (list && timeline) {
    const yearItems = Array.from(
      timeline.querySelectorAll(".ucm-blog-timeline__year")
    );
    const yearSections = Array.from(
      list.querySelectorAll(".ucm-blog-year-block")
    );
    const clock = timeline.querySelector(".ucm-blog-clock");
    let ticked = false;

    function isVisible(section) {
      return getComputedStyle(section).display !== "none";
    }

    function updateTimeline() {
      if (!yearSections.length) {
        return;
      }

      const threshold = window.scrollY + 160;
      let activeYear = null;

      yearSections.forEach(function (section) {
        if (
          isVisible(section) &&
          section.getBoundingClientRect().top + window.scrollY <= threshold
        ) {
          activeYear = section;
        }
      });

      if (!activeYear) {
        activeYear = yearSections.find(isVisible) || null;
      }

      let activeMonth = null;
      if (activeYear) {
        Array.from(activeYear.querySelectorAll(".ucm-blog-month")).forEach(
          function (month) {
            if (
              isVisible(month) &&
              month.getBoundingClientRect().top + window.scrollY <= threshold
            ) {
              activeMonth = month;
            }
          }
        );
      }

      yearItems.forEach(function (item) {
        const link = item.querySelector(".ucm-blog-timeline__year-link");
        const hash = link ? link.hash : "";
        item.classList.toggle(
          "is-active",
          Boolean(activeYear && hash === "#" + activeYear.id)
        );
      });

      timeline
        .querySelectorAll(".ucm-blog-timeline__months a")
        .forEach(function (link) {
          link.classList.toggle(
            "is-active",
            Boolean(activeMonth && link.hash === "#" + activeMonth.id)
          );
        });
    }

    let clockSeconds = 0;
    let lastClockFrame = 0;

    function updateClock(now) {
      if (!clock) {
        return;
      }

      if (!lastClockFrame) {
        lastClockFrame = now;
      }
      clockSeconds += Math.max(
        0,
        Math.min(0.5, (now - lastClockFrame) / 1000)
      );
      lastClockFrame = now;

      const max = document.documentElement.scrollHeight - window.innerHeight;
      const progress = max > 0 ? window.scrollY / max : 0;
      clock.style.setProperty(
        "--ucm-hour-angle",
        (305 - progress * 195 + clockSeconds * 0.36).toFixed(1) + "deg"
      );
      clock.style.setProperty(
        "--ucm-minute-angle",
        (60 + progress * 2340 + clockSeconds * 4.32).toFixed(1) + "deg"
      );
    }

    function animateClock(now) {
      updateClock(now);
      window.requestAnimationFrame(animateClock);
    }

    function onScroll() {
      if (ticked) {
        return;
      }
      ticked = true;
      window.requestAnimationFrame(function () {
        updateTimeline();
        ticked = false;
      });
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    updateTimeline();
    if (clock) {
      window.requestAnimationFrame(animateClock);
    }
  }

  const blogSidebar = document.querySelector(".ucm-blog-sidebar");
  const navOpen = document.querySelector("[data-blog-nav-open]");

  if (blogSidebar && navOpen) {
    navOpen.addEventListener("click", function () {
      blogSidebar.classList.add("is-nav-open");
    });

    const drawer = document.getElementById("__drawer");
    if (drawer) {
      drawer.addEventListener("change", function () {
        if (!drawer.checked) {
          blogSidebar.classList.remove("is-nav-open");
        }
      });
    }
  }

  const back = document.querySelector("[data-blog-back]");

  if (back) {
    back.addEventListener("click", function (event) {
      if (!document.referrer) {
        return;
      }
      try {
        const referrer = new URL(document.referrer);
        if (
          referrer.origin === window.location.origin &&
          /\/blog\/$/.test(referrer.pathname)
        ) {
          event.preventDefault();
          window.location.assign(referrer.href);
        }
      } catch (error) {
        /* A malformed referrer is not a reason to block the default link. */
      }
    });
  }
})();
