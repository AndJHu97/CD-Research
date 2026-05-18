import sys

def fix_line(line):
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return line

    # Only apply to coordinate region lines that are malformed
    if len(line) < 46:
        return line

    # insert space after column 46
    return line[:46] + " " + line[46:]


def fix_file(inp, out):
    with open(inp) as f_in, open(out, "w") as f_out:
        for line in f_in:
            f_out.write(fix_line(line))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_pdb_space.py input.pdb output.pdb")
        sys.exit(1)

    fix_file(sys.argv[1], sys.argv[2])
    print("Done.")