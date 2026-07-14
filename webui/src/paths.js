// Windows Explorer's "Copy as path" wraps the path in quotes:
//   "C:\models\ControlNet"
// Pasted verbatim, every downstream os.path call misses. The backend strips
// these too (config_io.strip_path_quotes) — doing it here as well means the
// field visibly cleans itself instead of failing silently later.
export function stripQuotes(value) {
  let cleaned = String(value ?? '').trim();
  while (cleaned.length >= 2 && cleaned[0] === cleaned.at(-1) && (cleaned[0] === '"' || cleaned[0] === "'")) {
    cleaned = cleaned.slice(1, -1).trim();
  }
  return cleaned;
}

// Field names that hold a filesystem path (mirrors the backend's heuristic).
export function isPathField(name) {
  return ['path', 'dir', 'folder'].some((hint) => name.includes(hint));
}
