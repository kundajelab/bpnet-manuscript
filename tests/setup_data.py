"""Creates a small test dataset
"""
import pyBigWig

ddir = "tests/data"

bw_pos = pyBigWig.open(f"{ddir}/pos.bw", "w")
bw_pos.addHeader([("chr1", 100), ("chr2", 100)])
bw_pos.addEntries("chr1", [10, 30, 80], values=[10.0, 150.0, 25.0], span=20)
bw_pos.addEntries("chr2", [10, 30, 80], values=[10.0, 150.0, 25.0], span=20)
bw_pos.close()


bw_neg = pyBigWig.open(f"{ddir}/neg.bw", "w")
bw_neg.addHeader([("chr1", 100), ("chr2", 100)])
bw_neg.addEntries("chr1", [15, 35, 85], values=[10.0, 150.0, 25.0], span=10)
bw_neg.addEntries("chr2", [15, 35, 85], values=[10.0, 150.0, 25.0], span=10)
bw_neg.close()

with open(f"{ddir}/peaks.bed", 'w') as f:
    f.write("chr1\t10\t30\nchr1\t40\t50\n")
    f.write("chr2\t10\t30\nchr2\t40\t50\n")

seq = "ACGTG" * 20
with open(f"{ddir}/ref.fa", 'w') as f:
    f.write("> chr1\n" + seq + "\n")
    f.write("> chr2\n" + seq + "\n")
