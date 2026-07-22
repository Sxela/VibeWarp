const plainObject = (value) => value !== null && typeof value === 'object'
  && !Array.isArray(value);

export function diffSettings(left, right, path = '', rows = []) {
  if (JSON.stringify(left) === JSON.stringify(right)) return rows;
  if (plainObject(left) && plainObject(right)) {
    let keys = [...new Set([...Object.keys(left), ...Object.keys(right)])].sort();
    for (let key of keys) {
      diffSettings(left[key], right[key], path ? `${path}.${key}` : key, rows);
    }
    return rows;
  }
  rows.push({path: path || '(root)', left, right});
  return rows;
}

export function formatSettingValue(value) {
  if (value === undefined) return '—';
  if (value === '') return '""';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}
