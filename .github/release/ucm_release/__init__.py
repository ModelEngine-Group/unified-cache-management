"""Compact UCM release package and ``python -m ucm_release`` dispatcher."""

from __future__ import annotations

import sys

# Python imports a package before looking for package.__main__.  Dispatch while
# argv[0] is still ``-m`` so the compact package does not spend one of its eight
# production-file slots on a forwarding-only __main__.py.
if sys.argv and sys.argv[0] == "-m":
    from .cli import main

    raise SystemExit(main())
