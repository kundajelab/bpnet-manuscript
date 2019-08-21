"""Taken from 

https://github.com/biopython/biopython/issues/232#issuecomment-326682150

LICENSE
Biopython is currently released under the "Biopython License Agreement" (given in full below). Unless stated otherwise in individual file headers, all Biopython's files are under the "Biopython License Agreement".

Some files are explicitly dual licensed under your choice of the "Biopython License Agreement" or the "BSD 3-Clause License" (both given in full below). This is with the intention of later offering all of Biopython under this dual licensing approach.

Biopython License Agreement
Permission to use, copy, modify, and distribute this software and its documentation with or without modifications and for any purpose and without fee is hereby granted, provided that any copyright notices appear in all copies and that both those copyright notices and this permission notice appear in supporting documentation, and that the names of the contributors or copyright holders not be used in advertising or publicity pertaining to distribution of the software without specific prior permission.

THE CONTRIBUTORS AND COPYRIGHT HOLDERS OF THIS SOFTWARE DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

BSD 3-Clause License
Copyright (c) 1999-2019, The Biopython Contributors All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from Bio.Alphabet import IUPAC
from Bio import Seq
from Bio import motifs
import math

def read(handle):
    """Parses the text output of the MEME program into a meme.Record object.
    Example:
    >>> from Bio.motifs import meme
    >>> with open("meme.output.txt") as f:
    ...     record = meme.read(f)
    >>> for motif in record:
    ...     for instance in motif.instances:
    ...         print(instance.motif_name, instance.sequence_name, instance.strand, instance.pvalue)
    """
    motif_number = 0
    record = Record()
    __read_version(record, handle)
    __read_alphabet(record, handle)
    __read_background(record, handle)

    while True:
        for line in handle:
            if line.startswith('MOTIF'):
                break
        else:
            return record
        name = line.split()[1]
        motif_number += 1
        length, num_occurrences, evalue = __read_motif_statistics(line, handle)
        counts = __read_lpm(line, handle)
        #{'A': 0.25, 'C': 0.25, 'T': 0.25, 'G': 0.25}
        motif = motifs.Motif(alphabet=record.alphabet, counts=counts)
        motif.background = record.background
        motif.length = length
        motif.num_occurrences = num_occurrences
        motif.evalue = evalue
        motif.name = name
        record.append(motif)
        assert len(record) == motif_number

    return record


class Record(list):
    """A class for holding the results of a MEME run.    """

    def __init__(self):
        """__init__ (self)"""
        self.version = ""
        self.datafile = ""
        self.command = ""
        self.alphabet = None
        self.background = {}
        self.sequences = []

    def __getitem__(self, key):
        if isinstance(key, str):
            for motif in self:
                if motif.name == key:
                    return motif
        else:
            return list.__getitem__(self, key)


# Everything below is private

def __read_background(record, handle):
    for line in handle:
        if line.startswith('Background letter frequencies'):
            break
    else:
        raise ValueError("Improper input file. File should contain a line starting background frequencies.")
    try:
        line = next(handle)
    except StopIteration:
        raise ValueError("Unexpected end of stream: Expected to find line starting background frequencies.")
    line = line.strip()
    ls = line.split()
    A, C, G, T = float(ls[1]), float(ls[3]), float(ls[5]), float(ls[7])
    record.background = {'A': A, 'C': C, 'G': G, 'T': T}


def __read_version(record, handle):
    for line in handle:
        if line.startswith('MEME version'):
            break
    else:
        raise ValueError("Improper input file. File should contain a line starting MEME version.")
    line = line.strip()
    ls = line.split()
    record.version = ls[2]

def __read_alphabet(record, handle):
    for line in handle:
        if line.startswith('ALPHABET'):
            break
    else:
        raise ValueError("Unexpected end of stream: Expected to find line starting with 'ALPHABET'")
    if not line.startswith('ALPHABET'):
        raise ValueError("Line does not start with 'ALPHABET':\n%s" % line)
    line = line.strip()
    line = line.replace('ALPHABET= ', '')
    if line == 'ACGT':
        al = IUPAC.unambiguous_dna
    else:
        al = IUPAC.protein
    record.alphabet = al

def __read_lpm(line, handle):
    counts = [[], [], [], []]
    for line in handle:
        freqs = line.split()
        if len(freqs) != 4:
            break
        counts[0].append(int(float(freqs[0]) * 1000000))
        counts[1].append(int(float(freqs[1]) * 1000000))
        counts[2].append(int(float(freqs[2]) * 1000000))
        counts[3].append(int(float(freqs[3]) * 1000000))
    c = {}
    c['A'] = counts[0]
    c['C'] = counts[1]
    c['G'] = counts[2]
    c['T'] = counts[3]
    return c

def __read_motif_statistics(line, handle):
    # minimal :
    # 	 letter-probability matrix: alength= 4 w= 19 nsites= 17 E= 4.1e-009
    for line in handle:
        if line.startswith('letter-probability matrix:'):
            break
    num_occurrences = int(line.split("nsites=")[1].split()[0])
    length = int(line.split("w=")[1].split()[0])
    evalue = float(line.split("E=")[1].split()[0])
    return length, num_occurrences, evalue


def __read_motif_name(handle):
    for line in handle:
        if 'sorted by position p-value' in line:
            break
    else:
        raise ValueError('Unexpected end of stream: Failed to find motif name')
    line = line.strip()
    words = line.split()
    name = " ".join(words[0:2])
    return name
