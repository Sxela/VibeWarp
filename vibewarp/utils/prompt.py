"""Prompt parsing and scheduling utilities."""


def parse_prompt(prompt):
    """Parse a prompt string into (text, weight) tuple.

    Supports formats:
        "a beautiful painting"         -> ("a beautiful painting", 1.0)
        "a beautiful painting:0.8"     -> ("a beautiful painting", 0.8)
        "https://example.com/img:0.5"  -> ("https://example.com/img", 0.5)
    """
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])
