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

/** Was this dotted path actually present in a run's saved settings? */
export function wasSaved(saved, path) {
  if (!saved || !path) return false;
  let node = saved;
  for (let key of path.split('.')) {
    if (!plainObject(node) || !(key in node)) return false;
    node = node[key];
  }
  return true;
}

/**
 * Mark which side of each diff row actually saved the setting.
 *
 * Defaults are filled in when a run's settings load, so a run that predates a
 * setting is indistinguishable from one that chose the default — the row reads
 * as agreement when really one run had no such option. This restores that.
 */
export function annotateUnsaved(rows, leftSaved, rightSaved) {
  return rows.map(row => ({
    ...row,
    leftSaved: wasSaved(leftSaved, row.path),
    rightSaved: wasSaved(rightSaved, row.path),
  }));
}

/**
 * Paths one run saved and the other did not, even where the loaded values
 * agree — "this run predates the setting" is worth seeing on its own.
 */
export function unsavedOnlyRows(leftSaved, rightSaved, rows = []) {
  let seen = new Set(rows.map(row => row.path));
  let out = [];
  const walk = (left, right, path) => {
    let keys = [...new Set([
      ...(plainObject(left) ? Object.keys(left) : []),
      ...(plainObject(right) ? Object.keys(right) : []),
    ])].sort();
    for (let key of keys) {
      let next = path ? `${path}.${key}` : key;
      let l = plainObject(left) ? left[key] : undefined;
      let r = plainObject(right) ? right[key] : undefined;
      // Recurse when a whole section is missing on one side too, so the rows
      // name the individual settings rather than just "contact_sheet".
      if ((plainObject(l) && plainObject(r))
          || (plainObject(l) && r === undefined)
          || (plainObject(r) && l === undefined)) {
        walk(l, r, next);
        continue;
      }
      let inLeft = plainObject(left) && key in left;
      let inRight = plainObject(right) && key in right;
      if (inLeft !== inRight && !seen.has(next)) {
        out.push({path: next, left: l, right: r,
                  leftSaved: inLeft, rightSaved: inRight, sameValue: true});
      }
    }
  };
  walk(leftSaved, rightSaved, '');
  return out;
}

export function formatSettingValue(value) {
  if (value === undefined) return '—';
  if (value === '') return '""';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}
