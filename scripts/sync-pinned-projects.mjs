import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const username = process.env.GITHUB_USERNAME || "shreyashkar-ml";
const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
const required = process.argv.includes("--required");
const outputPath = resolve(process.cwd(), "data/pinned-projects.json");

if (!token) {
  if (required) {
    throw new Error("GITHUB_TOKEN or GH_TOKEN is required for a pinned-project sync.");
  }

  console.warn("No GitHub token found; keeping data/pinned-projects.json as the local fallback.");
  process.exit(0);
}

const query = `
  query PinnedRepositories($login: String!) {
    user(login: $login) {
      pinnedItems(first: 4, types: REPOSITORY) {
        nodes {
          ... on Repository {
            name
            description
            url
            homepageUrl
            stargazerCount
            primaryLanguage { name }
          }
        }
      }
    }
  }
`;

const response = await fetch("https://api.github.com/graphql", {
  method: "POST",
  headers: {
    Accept: "application/vnd.github+json",
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    "User-Agent": "shreyashkar-ml-portfolio"
  },
  body: JSON.stringify({ query, variables: { login: username } })
});

if (!response.ok) {
  throw new Error(`GitHub GraphQL request failed with HTTP ${response.status}.`);
}

const payload = await response.json();
if (payload.errors?.length) {
  throw new Error(payload.errors.map((error) => error.message).join("; "));
}

const projects = payload.data?.user?.pinnedItems?.nodes?.filter(Boolean).map((project) => ({
  name: project.name,
  url: project.url,
  website: project.homepageUrl || undefined,
  description: project.description || "A pinned project from GitHub.",
  language: project.primaryLanguage?.name || "Open source",
  stars: project.stargazerCount,
  brandColor: "#ff90e8"
})) || [];

if (projects.length !== 4) {
  throw new Error(`Expected 4 pinned repositories, received ${projects.length}.`);
}

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify({ username, projects, syncedAt: new Date().toISOString() }, null, 2)}\n`);
console.log(`Synced ${projects.length} pinned repositories for ${username}.`);
