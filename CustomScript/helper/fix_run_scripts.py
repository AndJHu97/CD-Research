from pathlib import Path

base = Path(__file__).parent
runs = sorted(base.glob('run_*.sh'))
fixed = []
for p in runs:
    b = p.read_bytes()
    b2 = b.replace(b'\r\n', b'\n')
    # also convert lone CR if present
    b2 = b2.replace(b'\r', b'\n')
    p.write_bytes(b2)
    if b2.find(b'\r') == -1:
        fixed.append(p.name)

if fixed:
    print('Fixed: ' + ', '.join(fixed))
else:
    print('No run_*.sh files found or no changes needed')
