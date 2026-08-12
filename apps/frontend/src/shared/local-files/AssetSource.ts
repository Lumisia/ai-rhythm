export interface AssetSource {
  has(relativePath: string): boolean;
  readBytes(relativePath: string): Promise<ArrayBuffer>;
  readText(relativePath: string): Promise<string>;
}
