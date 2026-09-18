import fs from "node:fs/promises";
import path from "node:path";

const sourceRoot = path.resolve(process.env.SOURCE_ROOT || "");
const targetRoot = path.resolve(process.env.TARGET_ROOT || "");
if (!process.env.SOURCE_ROOT || !process.env.TARGET_ROOT || sourceRoot === targetRoot) {
  throw new Error("distinct SOURCE_ROOT and TARGET_ROOT are required");
}

const directoryPaths = ["docs/docs-site/docs/en", "docs/docs-site/docs/zh"];
const filePaths = ["docs/docs-site/translation-state.json"];

async function collect(root, relative, output) {
  const absolute = path.join(root, ...relative.split("/"));
  let entries;
  try {
    entries = await fs.readdir(absolute, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  for (const entry of entries) {
    const child = `${relative}/${entry.name}`;
    if (entry.isSymbolicLink()) throw new Error("documentation snapshot contains a symbolic link");
    if (entry.isDirectory()) await collect(root, child, output);
    else if (entry.isFile()) output.push(child);
    else throw new Error("documentation snapshot contains a non-file entry");
  }
}

const sourceFiles = [];
const targetFiles = [];
for (const relative of directoryPaths) {
  await collect(sourceRoot, relative, sourceFiles);
  await collect(targetRoot, relative, targetFiles);
}
for (const relative of filePaths) {
  for (const [root, output] of [[sourceRoot, sourceFiles], [targetRoot, targetFiles]]) {
    try {
      const stat = await fs.lstat(path.join(root, ...relative.split("/")));
      if (stat.isSymbolicLink() || !stat.isFile()) throw new Error("translation state is not a regular file");
      output.push(relative);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
}

const sourceSet = new Set(sourceFiles);
for (const relative of targetFiles) {
  if (!sourceSet.has(relative)) {
    await fs.unlink(path.join(targetRoot, ...relative.split("/")));
  }
}
for (const relative of sourceFiles) {
  const source = path.join(sourceRoot, ...relative.split("/"));
  const target = path.join(targetRoot, ...relative.split("/"));
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.copyFile(source, target);
  await fs.chmod(target, 0o644);
}
