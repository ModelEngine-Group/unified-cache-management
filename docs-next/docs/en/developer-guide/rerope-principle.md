# ReRoPE Principle

Design principle behind Rectified Rotary Position Embeddings (ReRoPE) — the algorithm rationale currently in `user-guide/capabilities/rerope.md`. The User Guide keeps only the quick-start / enable how-to.

!!! info "Under construction"

    This page will hold the ReRoPE algorithm rationale moved from
    `user-guide/capabilities/rerope.md`. The outline below describes what to write.

## Planned content

- **What ReRoPE does**: extending context length without fine-tuning by combining direct extrapolation with position interpolation; the window $w$, in-window interval 1 vs outside interval $1/k$, and the $k \to \infty$ limit.
- **Attention score formulas**: the two score paths (within-$w$ local, $\ge w$ global compressed), with the existing math.
- **Throughput trade-off**: double attention cost (local plus global) and how local windows balance it; value for training-free long context.
- **Triton implementation notes**: data loading (query2 with alternative rotary position, unrotated key2) and the rerope mask construction.
- **Pointers**: link to the quick-start in User Guide and to the blog posts.

## Do not

- Do not duplicate quick-start commands or env vars (User Guide).
- Keep the blog links as references.

## Acceptance

- A reader can explain the mechanism and the throughput cost.
- The math, the User Guide quick-start link, and the blog links are present.

Owner: ____
