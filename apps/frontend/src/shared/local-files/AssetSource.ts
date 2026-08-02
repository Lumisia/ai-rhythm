export interface AssetSource {
  readBytes(relativePath: string): Promise<ArrayBuffer>;
  readText(relativePath: string): Promise<string>;
}
