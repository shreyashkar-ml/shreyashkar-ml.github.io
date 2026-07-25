# Shreyashkar Lal Sahu — Portfolio

Live: https://shreyashkar-ml.github.io

A Hugo-based personal portfolio site.

- Theme: PaperMod (configured in hugo.toml: theme = "PaperMod")
- LaTeX / math support: enabled via the theme (params.math = true in hugo.toml) — math expressions are supported in content.
- Config: See hugo.toml for site params, menu and markup settings (Goldmark renderer settings are included).

Quick start (local preview)

1. Install Hugo.
2. Clone the repo and run:
   git clone https://github.com/shreyashkar-ml/shreyashkar-ml.github.io.git
   cd shreyashkar-ml.github.io
   hugo server -D
3. Open http://localhost:1313

## GitHub project sync

The homepage renders the four pinned repositories from `data/pinned-projects.json`. GitHub Pages refreshes this data automatically during the scheduled workflow using the GitHub GraphQL API, including repository descriptions, primary languages, and star counts.

For a local refresh, provide a GitHub token and run:

```bash
GITHUB_TOKEN=your_token node scripts/sync-pinned-projects.mjs
```

Without a token, local builds use the checked-in fallback snapshot.

Thanks for visiting!
