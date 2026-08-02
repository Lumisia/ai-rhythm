import type { AssetSource } from "./AssetSource";

function readFileBytes(file: File): Promise<ArrayBuffer> {
  if (typeof file.arrayBuffer === "function") {
    return file.arrayBuffer();
  }

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error(`failed to read ${file.name}`));
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        resolve(reader.result);
      } else {
        reject(new Error(`failed to read ${file.name} as bytes`));
      }
    };
    reader.readAsArrayBuffer(file);
  });
}

export function normalizeRelativePath(candidate: string): string {
  const normalized = candidate.replaceAll("\\", "/");
  if (!normalized || normalized.startsWith("/") || /^[a-zA-Z]:\//.test(normalized)) {
    throw new Error(`invalid relative path: ${candidate}`);
  }

  const segments = normalized.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    throw new Error(`invalid relative path: ${candidate}`);
  }
  return segments.join("/");
}

function pathInsideSelectedRoot(file: File): string {
  const browserPath = file.webkitRelativePath;
  if (!browserPath) {
    return normalizeRelativePath(file.name);
  }

  const fullPath = normalizeRelativePath(browserPath);
  const separator = fullPath.indexOf("/");
  if (separator < 0 || separator === fullPath.length - 1) {
    throw new Error(`invalid selected-directory path: ${browserPath}`);
  }
  return fullPath.slice(separator + 1);
}

export class LocalDirectoryAssetSource implements AssetSource {
  readonly #files = new Map<string, File>();
  readonly #canonicalPaths = new Map<string, string>();

  constructor(files: Iterable<File>) {
    for (const file of files) {
      const path = pathInsideSelectedRoot(file);
      const canonicalPath = path.toLocaleLowerCase("en-US");
      const existing = this.#canonicalPaths.get(canonicalPath);
      if (existing) {
        throw new Error(`duplicate path differs only by case: ${existing}, ${path}`);
      }
      this.#canonicalPaths.set(canonicalPath, path);
      this.#files.set(path, file);
    }
  }

  async readBytes(relativePath: string): Promise<ArrayBuffer> {
    const path = normalizeRelativePath(relativePath);
    const file = this.#files.get(path);
    if (!file) {
      throw new Error(`missing file: ${path}`);
    }
    return readFileBytes(file);
  }

  async readText(relativePath: string): Promise<string> {
    const bytes = await this.readBytes(relativePath);
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  }
}
