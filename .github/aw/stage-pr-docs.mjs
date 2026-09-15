import fs from "node:fs/promises";
import path from "node:path";

const apiUrl = process.env.GITHUB_API_URL || "https://api.github.com";
const token = process.env.GH_TOKEN;
const baseRepository = process.env.BASE_REPOSITORY;
const headRepository = process.env.HEAD_REPOSITORY;
const headSha = process.env.HEAD_SHA;
const pullNumber = Number(process.env.PR_NUMBER);
const workspace = path.resolve(process.env.GITHUB_WORKSPACE || process.cwd());
const maxFileBytes = 2 * 1024 * 1024;
const maxTotalBytes = 12 * 1024 * 1024;

if (!token || !baseRepository || !headRepository || !headSha || !pullNumber) {
  throw new Error("required pull request snapshot inputs are missing");
}
if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(baseRepository)) {
  throw new Error("invalid base repository name");
}
if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(headRepository)) {
  throw new Error("invalid head repository name");
}
if (!/^[0-9a-f]{40}$/.test(headSha)) {
  throw new Error("invalid pull request head SHA");
}

async function github(endpoint) {
  const response = await fetch(`${apiUrl}${endpoint}`, {
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
    },
  });
  if (!response.ok) {
    throw new Error(`GitHub API request failed with HTTP ${response.status}`);
  }
  return response.json();
}

function isAllowed(relative) {
  return (
    relative.startsWith("docs/docs-site/docs/en/") ||
    relative.startsWith("docs/docs-site/docs/zh/") ||
    relative === "docs/docs-site/translation-state.json"
  );
}

function checkedTarget(relative) {
  if (
    relative.includes("\\") ||
    relative.split("/").includes("..") ||
    path.posix.normalize(relative) !== relative ||
    !isAllowed(relative)
  ) {
    throw new Error("unsafe documentation path returned by GitHub");
  }
  const target = path.resolve(workspace, ...relative.split("/"));
  if (target !== workspace && !target.startsWith(`${workspace}${path.sep}`)) {
    throw new Error("documentation path escapes the workspace");
  }
  return target;
}

async function collectExisting(root) {
  const result = [];
  async function visit(current) {
    let entries;
    try {
      entries = await fs.readdir(current, { withFileTypes: true });
    } catch (error) {
      if (error.code === "ENOENT") return;
      throw error;
    }
    for (const entry of entries) {
      const absolute = path.join(current, entry.name);
      const relative = path.relative(workspace, absolute).split(path.sep).join("/");
      if (entry.isSymbolicLink()) {
        throw new Error("trusted documentation snapshot contains a symbolic link");
      }
      if (entry.isDirectory()) await visit(absolute);
      else if (entry.isFile() && isAllowed(relative)) result.push(relative);
      else if (!entry.isFile()) throw new Error("documentation snapshot contains a non-file entry");
    }
  }
  await visit(root);
  return result;
}

const [baseOwner, baseRepo] = baseRepository.split("/");
const pull = await github(`/repos/${baseOwner}/${baseRepo}/pulls/${pullNumber}`);
if (
  pull.state !== "open" ||
  String(pull.head.sha) !== headSha ||
  pull.head.repo.full_name !== headRepository ||
  pull.base.repo.full_name !== baseRepository
) {
  throw new Error("pull request metadata changed before documentation staging");
}

const [headOwner, headRepo] = headRepository.split("/");
const tree = await github(
  `/repos/${headOwner}/${headRepo}/git/trees/${headSha}?recursive=1`,
);
if (tree.truncated) {
  throw new Error("pull request tree is truncated; refusing a partial snapshot");
}

const blobs = new Map();
for (const entry of tree.tree || []) {
  if (!isAllowed(entry.path)) continue;
  checkedTarget(entry.path);
  if (entry.type === "tree" && entry.mode === "040000") continue;
  if (entry.type !== "blob" || !["100644", "100755"].includes(entry.mode)) {
    throw new Error("documentation input contains a symlink or non-file entry");
  }
  if (!entry.sha || !Number.isInteger(entry.size) || entry.size > maxFileBytes) {
    throw new Error("documentation input exceeds the per-file size limit");
  }
  blobs.set(entry.path, entry);
}

const existing = [
  ...(await collectExisting(path.join(workspace, "docs", "docs-site", "docs", "en"))),
  ...(await collectExisting(path.join(workspace, "docs", "docs-site", "docs", "zh"))),
];
try {
  const state = await fs.lstat(path.join(workspace, "docs", "docs-site", "translation-state.json"));
  if (state.isSymbolicLink() || !state.isFile()) {
    throw new Error("trusted translation state is not a regular file");
  }
  existing.push("docs/docs-site/translation-state.json");
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

for (const relative of existing) {
  if (!blobs.has(relative)) await fs.unlink(checkedTarget(relative));
}

let totalBytes = 0;
const decoder = new TextDecoder("utf-8", { fatal: true });
for (const [relative, entry] of [...blobs].sort(([left], [right]) => left.localeCompare(right))) {
  const blob = await github(`/repos/${headOwner}/${headRepo}/git/blobs/${entry.sha}`);
  if (blob.encoding !== "base64" || typeof blob.content !== "string") {
    throw new Error("GitHub returned an unsupported documentation blob encoding");
  }
  const bytes = Buffer.from(blob.content.replace(/\n/g, ""), "base64");
  if (bytes.length !== entry.size) throw new Error("documentation blob size changed");
  totalBytes += bytes.length;
  if (totalBytes > maxTotalBytes) throw new Error("documentation snapshot exceeds the size limit");
  decoder.decode(bytes);
  const target = checkedTarget(relative);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, bytes, { mode: 0o644 });
}
