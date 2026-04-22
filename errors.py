import re
from collections import Counter

log_file = "logs.txt"          # input log
output_file = "unique_errors.txt"

error_blocks = []
current_error = []

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Detect start of an error
        if "[ERROR" in line or line.startswith("Traceback (most recent call last):"):
            if current_error:
                error_blocks.append("".join(current_error))
                current_error = []
            current_error.append(line)
        elif current_error:
            # Continue collecting traceback lines
            current_error.append(line)

    # Catch last error
    if current_error:
        error_blocks.append("".join(current_error))

def normalize_error(err: str) -> str:
    """Remove timestamps, frame numbers, and paths"""
    err = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+", "", err)   # timestamps
    err = re.sub(r"-Frame\s+\d+-", "", err)              # frame numbers
    err = re.sub(r"/home/[^ ]+", "<PATH>", err)          # file paths
    return err.strip()

normalized_errors = [normalize_error(e) for e in error_blocks]
counts = Counter(normalized_errors)

with open(output_file, "w", encoding="utf-8") as out:
    for error, count in counts.items():
        out.write(f"COUNT: {count}\n")
        out.write(error + "\n")
        out.write("=" * 80 + "\n")

print(f"Saved {len(counts)} unique errors to {output_file}")
