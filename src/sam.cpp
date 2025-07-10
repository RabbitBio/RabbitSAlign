#include "sam.hpp"
#include <algorithm>
#include <ostream>
#include <sstream>
#include <iostream>

#define SAM_UNMAPPED_MAPQ 0
#define SAM_UNMAPPED_MAPQ_STRING "0"

using namespace klibpp;

/*
 * SAM columns
 *
 * 1 QNAME query template name
 * 2 FLAG  bitwise flag
 * 3 RNAME reference sequence name
 * 4 POS   1-based leftmost mapping position (set to 0 for unmapped read w/o pos)
 * 5 MAPQ  mapping quality (255 if unavailable)
 * 6 CIGAR
 * 7 RNEXT reference name of mate/next read
 * 8 PNEXT position of mate/next read
 * 9 TLEN template length
 * 10 SEQ
 * 11 QUAL
 * 12 optional fields
*/

/* Strip the /1 or /2 suffix from a read name */
std::string strip_suffix(const std::string& name) {
    auto len = name.length();
    if (
        len >= 2
        && name[len - 2] == '/'
        && (name[len - 1] == '1' || name[len - 1] == '2')
    ) {
        return std::string{name, 0, len - 2};
    } else {
        return name;
    }
}

std::string strip_suffix(const char* name, int len) {
    if (
        len >= 2
        && name[len - 2] == '/'
        && (name[len - 1] == '1' || name[len - 1] == '2')
    ) {
        return std::string(name, len - 2);
    } else {
        return std::string(name, len);
    }
}

void Sam::append_rg_and_newline() {
    sam_string.append(tail);
}

void Sam::append_details(const Details& details) {
    std::stringstream s;
    s << "\tna:i:" << details.nams;
    s << "\tnr:i:" << details.nam_rescue;
    s << "\tal:i:" << details.tried_alignment;
    s << "\tga:i:" << details.gapped;
    sam_string.append(s.str());
}

void Sam::append_paired_details(const Details& details) {
    std::stringstream s;
    s << "\tmr:i:" << details.mate_rescue;
    sam_string.append(s.str());
}

std::string Sam::cigar_string(const CigarData& cigar) const {
    int n_cigar = cigar.cigar[0];
    if (n_cigar == 0) {
        // This case should normally not occur because
        // unmapped reads would be added with add_unmapped,
        // which hardcodes the "*"
        return "*";
    }
    if (cigar_ops == CigarOps::EQX) {
        std::stringstream s;
        for (int i = 0; i < n_cigar; ++i) {
            uint32_t op_len = cigar.cigar[i + 1];
            s << (op_len >> 4) << "MIDNSHP=X"[op_len & 0xf];
        }
        return s.str();
    } else {
        Cigar cigar2;
        for (int i = 0; i < n_cigar; i++) {
            uint32_t op_len = cigar.cigar[i + 1];
            auto op = op_len & 0xf;
            auto len = op_len >> 4;
            if (op == CIGAR_EQ || op == CIGAR_X) {
                cigar2.push(CIGAR_MATCH, len);
            } else {
                cigar2.push(op, len);
            }
        }
        return cigar2.to_string();
    }
}

std::string Sam::cigar_string(const Cigar& cigar) const {
    if (cigar.empty()) {
        // This case should normally not occur because
        // unmapped reads would be added with add_unmapped,
        // which hardcodes the "*"
        return "*";
    }
    if (cigar_ops == CigarOps::EQX) {
        return cigar.to_string();
    } else {
        return cigar.to_m().to_string();
    }
}

void Sam::add_unmapped(const neoReference& record, uint16_t flags) {
    if (!output_unmapped) {
        return;
    }
    assert((flags & ~(UNMAP|PAIRED|MUNMAP|READ1|READ2)) == 0);
    assert(flags & UNMAP);
    sam_string.append(strip_suffix((char*)record.base + record.pname, record.lname));
    sam_string.append("\t");
    sam_string.append(std::to_string(flags));
    sam_string.append("\t*\t0\t" SAM_UNMAPPED_MAPQ_STRING "\t*\t*\t0\t0\t");
    append_seq((char*)record.base + record.pseq, record.lseq);
    append_qual((char*)record.base + record.pqual, record.lqual);
    append_rg_and_newline();
}

void Sam::add_unmapped(const KSeq& record, uint16_t flags) {
    if (!output_unmapped) {
        return;
    }
    assert((flags & ~(UNMAP|PAIRED|MUNMAP|READ1|READ2)) == 0);
    assert(flags & UNMAP);
    sam_string.append(strip_suffix(record.name));
    sam_string.append("\t");
    sam_string.append(std::to_string(flags));
    sam_string.append("\t*\t0\t" SAM_UNMAPPED_MAPQ_STRING "\t*\t*\t0\t0\t");
    append_seq(record.seq);
    append_qual(record.qual);
    append_rg_and_newline();
}

void Sam::add_unmapped_mate(const neoReference& record, uint16_t flags, const std::string& mate_reference_name, uint32_t mate_pos) {
    assert(flags & (UNMAP|PAIRED));
    sam_string.append(strip_suffix((char*)record.base + record.pname, record.lname));
    sam_string.append("\t");
    sam_string.append(std::to_string(flags));
    sam_string.append("\t");
    // The SAM specification recommends: "For a unmapped paired-end or
    // mate-pair read whose mate is mapped, the unmapped read should have
    // RNAME and POS identical to its mate."
    sam_string.append(mate_reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(mate_pos + 1));
    sam_string.append("\t" SAM_UNMAPPED_MAPQ_STRING "\t*\t");
    sam_string.append("=");
    sam_string.append("\t");
    sam_string.append(std::to_string(mate_pos + 1));
    sam_string.append("\t0\t");
    append_seq((char*)record.base + record.pseq, record.lseq);
    append_qual((char*)record.base + record.pqual, record.lqual);
    append_rg_and_newline();
}

void Sam::add_unmapped_mate(const KSeq& record, uint16_t flags, const std::string& mate_reference_name, uint32_t mate_pos) {
    assert(flags & (UNMAP|PAIRED));
    sam_string.append(strip_suffix(record.name));
    sam_string.append("\t");
    sam_string.append(std::to_string(flags));
    sam_string.append("\t");
    // The SAM specification recommends: "For a unmapped paired-end or
    // mate-pair read whose mate is mapped, the unmapped read should have
    // RNAME and POS identical to its mate."
    sam_string.append(mate_reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(mate_pos + 1));
    sam_string.append("\t" SAM_UNMAPPED_MAPQ_STRING "\t*\t");
    sam_string.append("=");
    sam_string.append("\t");
    sam_string.append(std::to_string(mate_pos + 1));
    sam_string.append("\t0\t");
    append_seq(record.seq);
    append_qual(record.qual);
    append_rg_and_newline();
}

void Sam::add_unmapped_pair(const KSeq& r1, const KSeq& r2) {
    add_unmapped(r1, PAIRED | UNMAP | MUNMAP | READ1);
    add_unmapped(r2, PAIRED | UNMAP | MUNMAP | READ2);
}

void Sam::add_unmapped_pair(const neoReference& r1, const neoReference& r2) {
    add_unmapped(r1, PAIRED | UNMAP | MUNMAP | READ1);
    add_unmapped(r2, PAIRED | UNMAP | MUNMAP | READ2);
}

// Add single-end alignment
void Sam::add(
    const Alignment& alignment,
    const KSeq& record,
    const std::string& sequence_rc,
    uint8_t mapq,
    bool is_primary,
    const Details& details
) {
    assert(!alignment.is_unaligned);

    int flags = 0;
    if (!alignment.is_unaligned && alignment.is_rc) {
        flags |= REVERSE;
    }
    if (!is_primary) {
        flags |= SECONDARY;
        mapq = 255;
    }
    add_record(record.name, flags, references.names[alignment.ref_id], alignment.ref_start, mapq, alignment.cigar, "*", -1, 0, record.seq, sequence_rc, record.qual, alignment.edit_distance, alignment.score, details);
}

void Sam::add(
        const std::pair<GPUAlignment, CigarData>& alignment,
        const neoReference& record,
        const char* sequence_rc,
        uint8_t mapq,
        bool is_primary,
        const Details& details
) {
    assert(!alignment.first.is_unaligned);

    int flags = 0;
    if (!alignment.first.is_unaligned && alignment.first.is_rc) {
        flags |= REVERSE;
    }
    if (!is_primary) {
        flags |= SECONDARY;
        mapq = 255;
    }
    add_record((char*)record.base + record.pname, record.lname, flags, references.names[alignment.first.ref_id], alignment.first.ref_start, mapq, alignment.second, "*", -1, 0,
               (char*)record.base + record.pseq, sequence_rc, record.lseq, (char*)record.base + record.pqual, record.lqual, alignment.first.edit_distance, alignment.first.score, details);
}

void Sam::add(
    const std::pair<GPUAlignment, CigarData>& alignment,
    const KSeq& record,
    const std::string& sequence_rc,
    uint8_t mapq,
    bool is_primary,
    const Details& details
) {
    assert(!alignment.first.is_unaligned);

    int flags = 0;
    if (!alignment.first.is_unaligned && alignment.first.is_rc) {
        flags |= REVERSE;
    }
    if (!is_primary) {
        flags |= SECONDARY;
        mapq = 255;
    }
    add_record(record.name, flags, references.names[alignment.first.ref_id], alignment.first.ref_start, mapq, alignment.second, "*", -1, 0, record.seq, sequence_rc, record.qual, alignment.first.edit_distance, alignment.first.score, details);
}

// Add one individual record
void Sam::add_record(
    const std::string& query_name,
    uint16_t flags,
    const std::string& reference_name,
    uint32_t pos,
    uint8_t mapq,
    const Cigar& cigar,
    const std::string& mate_reference_name,
    uint32_t mate_pos,
    int32_t template_len,
    const std::string& query_sequence,
    const std::string& query_sequence_rc,
    const std::string& qual,
    int ed,
    int aln_score,
    const Details& details
) {
    sam_string.append(strip_suffix(query_name));
    sam_string.append("\t");
    sam_string.append(std::to_string(flags));
    sam_string.append("\t");
    sam_string.append(reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(pos + 1));  // convert to 1-based coordinate
    sam_string.append("\t");
    sam_string.append(std::to_string(mapq));
    sam_string.append("\t");
    sam_string.append(cigar_string(cigar));
    sam_string.append("\t");

    sam_string.append(mate_reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(mate_pos + 1));
    sam_string.append("\t");
    sam_string.append(std::to_string(template_len));
    sam_string.append("\t");

    if (flags & SECONDARY) {
        append_seq("");
    } else if (flags & REVERSE) {
        append_seq(query_sequence_rc);
    } else {
        append_seq(query_sequence);
    }

    if (!(flags & UNMAP)) {
        if (flags & SECONDARY) {
            append_qual("");
        } else if (flags & REVERSE) {
            auto qual_rev = qual;
            std::reverse(qual_rev.begin(), qual_rev.end());
            append_qual(qual_rev);
        } else {
            append_qual(qual);
        }
        sam_string.append("\t");
        sam_string.append("NM:i:");
        sam_string.append(std::to_string(ed));
        sam_string.append("\t");
        sam_string.append("AS:i:");
        sam_string.append(std::to_string(aln_score));
    } else {
        append_qual(qual);
    }

    if (show_details) {
        append_details(details);
        if (flags & PAIRED) {
            append_paired_details(details);
        }
    }
    append_rg_and_newline();
}

void Sam::add_record(
    const char* query_name,
    int name_len,
    uint16_t flags,
    const std::string& reference_name,
    uint32_t pos,
    uint8_t mapq,
    const CigarData& cigar,
    const std::string& mate_reference_name,
    uint32_t mate_pos,
    int32_t template_len,
    const char* query_sequence,
    const char* query_sequence_rc,
    int seq_len,
    const char* qual,
    int qual_len,
    int ed,
    int aln_score,
    const Details& details
) {
    sam_string.append(strip_suffix(query_name, name_len));
    sam_string.append("\t");
    sam_string.append(std::to_string(flags));
    sam_string.append("\t");
    sam_string.append(reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(pos + 1));  // convert to 1-based coordinate
    sam_string.append("\t");
    sam_string.append(std::to_string(mapq));
    sam_string.append("\t");
    sam_string.append(cigar_string(cigar));
    sam_string.append("\t");

    sam_string.append(mate_reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(mate_pos + 1));
    sam_string.append("\t");
    sam_string.append(std::to_string(template_len));
    sam_string.append("\t");

    if (flags & SECONDARY) {
        append_seq("");
    } else if (flags & REVERSE) {
        append_seq(query_sequence_rc, seq_len);
    } else {
        append_seq(query_sequence, seq_len);
    }

    if (!(flags & UNMAP)) {
        if (flags & SECONDARY) {
            append_qual("");
        } else if (flags & REVERSE) {
            auto qual_rev = std::string(qual, qual_len);
            std::reverse(qual_rev.begin(), qual_rev.end());
            append_qual(qual_rev);
        } else {
            append_qual(qual, qual_len);
        }
        sam_string.append("\t");
        sam_string.append("NM:i:");
        sam_string.append(std::to_string(ed));
        sam_string.append("\t");
        sam_string.append("AS:i:");
        sam_string.append(std::to_string(aln_score));
    } else {
        append_qual(qual, qual_len);
    }

    if (show_details) {
        append_details(details);
        if (flags & PAIRED) {
            append_paired_details(details);
        }
    }
    append_rg_and_newline();
}

void Sam::add_record(
    const std::string& query_name,
    uint16_t flags,
    const std::string& reference_name,
    uint32_t pos,
    uint8_t mapq,
    const CigarData& cigar,
    const std::string& mate_reference_name,
    uint32_t mate_pos,
    int32_t template_len,
    const std::string& query_sequence,
    const std::string& query_sequence_rc,
    const std::string& qual,
    int ed,
    int aln_score,
    const Details& details
) {
    sam_string.append(strip_suffix(query_name));
    sam_string.append("\t");
    sam_string.append(std::to_string(flags));
    sam_string.append("\t");
    sam_string.append(reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(pos + 1));  // convert to 1-based coordinate
    sam_string.append("\t");
    sam_string.append(std::to_string(mapq));
    sam_string.append("\t");
    sam_string.append(cigar_string(cigar));
    sam_string.append("\t");

    sam_string.append(mate_reference_name);
    sam_string.append("\t");
    sam_string.append(std::to_string(mate_pos + 1));
    sam_string.append("\t");
    sam_string.append(std::to_string(template_len));
    sam_string.append("\t");

    if (flags & SECONDARY) {
        append_seq("");
    } else if (flags & REVERSE) {
        append_seq(query_sequence_rc);
    } else {
        append_seq(query_sequence);
    }

    if (!(flags & UNMAP)) {
        if (flags & SECONDARY) {
            append_qual("");
        } else if (flags & REVERSE) {
            auto qual_rev = qual;
            std::reverse(qual_rev.begin(), qual_rev.end());
            append_qual(qual_rev);
        } else {
            append_qual(qual);
        }
        sam_string.append("\t");
        sam_string.append("NM:i:");
        sam_string.append(std::to_string(ed));
        sam_string.append("\t");
        sam_string.append("AS:i:");
        sam_string.append(std::to_string(aln_score));
    } else {
        append_qual(qual);
    }

    if (show_details) {
        append_details(details);
        if (flags & PAIRED) {
            append_paired_details(details);
        }
    }
    append_rg_and_newline();
}

void Sam::add_pair(
    const std::pair<GPUAlignment, CigarData> &alignment1,
    const std::pair<GPUAlignment, CigarData> &alignment2,
    const neoReference& record1,
    const neoReference& record2,
    const char* read1_rc,
    const char* read2_rc,
    uint8_t mapq1,
    uint8_t mapq2,
    bool is_proper,
    bool is_primary,
    const std::array<Details, 2>& details
) {
    int f1 = PAIRED | READ1;
    int f2 = PAIRED | READ2;
    if (!is_primary) {
        f1 |= SECONDARY;
        f2 |= SECONDARY;
    }

    int template_len1 = 0;
    bool both_aligned = !alignment1.first.is_unaligned && !alignment2.first.is_unaligned;
    if (both_aligned && alignment1.first.ref_id == alignment2.first.ref_id) {
        const int dist = alignment2.first.ref_start - alignment1.first.ref_start;
        if (dist > 0) {
            template_len1 = dist + alignment2.first.length;
        }
        else {
            template_len1 = dist - alignment1.first.length;
        }
    }
    if (is_proper) {
        f1 |= PROPER_PAIR;
        f2 |= PROPER_PAIR;
    }

    std::string reference_name1;
    int pos1 = alignment1.first.ref_start;
    int edit_distance1 = alignment1.first.edit_distance;
    if (alignment1.first.is_unaligned) {
        f1 |= UNMAP;
        f2 |= MUNMAP;
        pos1 = -1;
        reference_name1 = "*";
    } else {
        if (alignment1.first.is_rc) {
            f1 |= REVERSE;
            f2 |= MREVERSE;
        }
        reference_name1 = references.names[alignment1.first.ref_id];
    }

    std::string reference_name2;
    int pos2 = alignment2.first.ref_start;
    int edit_distance2 = alignment2.first.edit_distance;
    if (alignment2.first.is_unaligned) {
        f2 |= UNMAP;
        f1 |= MUNMAP;
        pos2 = -1;
        reference_name2 = "*";
    } else {
        if (alignment2.first.is_rc) {
            f1 |= MREVERSE;
            f2 |= REVERSE;
        }
        reference_name2 = references.names[alignment2.first.ref_id];
    }

    // Reference name as used in the RNEXT field;
    // set to "=" if identical to reference_name
    std::string mate_reference_name1 = reference_name1;
    std::string mate_reference_name2 = reference_name2;
    if (
        (!alignment1.first.is_unaligned && !alignment2.first.is_unaligned && alignment1.first.ref_id == alignment2.first.ref_id)
        || (alignment1.first.is_unaligned != alignment2.first.is_unaligned)
    ) {
        mate_reference_name1 = "=";
        mate_reference_name2 = "=";
    }

    if (alignment1.first.is_unaligned != alignment2.first.is_unaligned) {
        if (alignment1.first.is_unaligned) {
            pos1 = pos2;
        } else {
            pos2 = pos1;
        }
    }

    if (alignment1.first.is_unaligned) {
        add_unmapped_mate(record1, f1, reference_name2, pos2);
    } else {
        add_record((char*)record1.base + record1.pname, record1.lname, f1, reference_name1, alignment1.first.ref_start, mapq1, alignment1.second, mate_reference_name2, pos2, template_len1,
                   (char*)record1.base + record1.pseq, read1_rc, record1.lseq, (char*)record1.base + record1.pqual, record1.lqual, edit_distance1, alignment1.first.score, details[0]);
    }
    if (alignment2.first.is_unaligned) {
        add_unmapped_mate(record2, f2, reference_name1, pos1);
    } else {
        add_record((char*)record2.base + record2.pname, record2.lname, f2, reference_name2, alignment2.first.ref_start, mapq2, alignment2.second, mate_reference_name1, pos1, -template_len1,
                   (char*)record2.base + record2.pseq, read2_rc, record2.lseq, (char*)record2.base + record2.pqual, record2.lqual, edit_distance2, alignment2.first.score, details[1]);

    }
}

void Sam::add_pair(
    const std::pair<GPUAlignment, CigarData> &alignment1,
    const std::pair<GPUAlignment, CigarData> &alignment2,
    const KSeq& record1,
    const KSeq& record2,
    const std::string &read1_rc,
    const std::string &read2_rc,
    uint8_t mapq1,
    uint8_t mapq2,
    bool is_proper,
    bool is_primary,
    const std::array<Details, 2>& details
) {
    int f1 = PAIRED | READ1;
    int f2 = PAIRED | READ2;
    if (!is_primary) {
        f1 |= SECONDARY;
        f2 |= SECONDARY;
    }

    int template_len1 = 0;
    bool both_aligned = !alignment1.first.is_unaligned && !alignment2.first.is_unaligned;
    if (both_aligned && alignment1.first.ref_id == alignment2.first.ref_id) {
        const int dist = alignment2.first.ref_start - alignment1.first.ref_start;
        if (dist > 0) {
            template_len1 = dist + alignment2.first.length;
        }
        else {
            template_len1 = dist - alignment1.first.length;
        }
    }
    if (is_proper) {
        f1 |= PROPER_PAIR;
        f2 |= PROPER_PAIR;
    }

    std::string reference_name1;
    int pos1 = alignment1.first.ref_start;
    int edit_distance1 = alignment1.first.edit_distance;
    if (alignment1.first.is_unaligned) {
        f1 |= UNMAP;
        f2 |= MUNMAP;
        pos1 = -1;
        reference_name1 = "*";
    } else {
        if (alignment1.first.is_rc) {
            f1 |= REVERSE;
            f2 |= MREVERSE;
        }
        reference_name1 = references.names[alignment1.first.ref_id];
    }

    std::string reference_name2;
    int pos2 = alignment2.first.ref_start;
    int edit_distance2 = alignment2.first.edit_distance;
    if (alignment2.first.is_unaligned) {
        f2 |= UNMAP;
        f1 |= MUNMAP;
        pos2 = -1;
        reference_name2 = "*";
    } else {
        if (alignment2.first.is_rc) {
            f1 |= MREVERSE;
            f2 |= REVERSE;
        }
        reference_name2 = references.names[alignment2.first.ref_id];
    }

    // Reference name as used in the RNEXT field;
    // set to "=" if identical to reference_name
    std::string mate_reference_name1 = reference_name1;
    std::string mate_reference_name2 = reference_name2;
    if (
        (!alignment1.first.is_unaligned && !alignment2.first.is_unaligned && alignment1.first.ref_id == alignment2.first.ref_id)
        || (alignment1.first.is_unaligned != alignment2.first.is_unaligned)
    ) {
        mate_reference_name1 = "=";
        mate_reference_name2 = "=";
    }

    if (alignment1.first.is_unaligned != alignment2.first.is_unaligned) {
        if (alignment1.first.is_unaligned) {
            pos1 = pos2;
        } else {
            pos2 = pos1;
        }
    }

    if (alignment1.first.is_unaligned) {
        add_unmapped_mate(record1, f1, reference_name2, pos2);
    } else {
        add_record(record1.name, f1, reference_name1, alignment1.first.ref_start, mapq1, alignment1.second, mate_reference_name2, pos2, template_len1, record1.seq, read1_rc, record1.qual, edit_distance1, alignment1.first.score, details[0]);
    }
    if (alignment2.first.is_unaligned) {
        add_unmapped_mate(record2, f2, reference_name1, pos1);
    } else {
        add_record(record2.name, f2, reference_name2, alignment2.first.ref_start, mapq2, alignment2.second, mate_reference_name1, pos1, -template_len1, record2.seq, read2_rc, record2.qual, edit_distance2, alignment2.first.score, details[1]);
    }
}


void Sam::add_pair(
    const Alignment &alignment1,
    const Alignment &alignment2,
    const KSeq& record1,
    const KSeq& record2,
    const std::string &read1_rc,
    const std::string &read2_rc,
    uint8_t mapq1,
    uint8_t mapq2,
    bool is_proper,
    bool is_primary,
    const std::array<Details, 2>& details
) {
    int f1 = PAIRED | READ1;
    int f2 = PAIRED | READ2;
    if (!is_primary) {
        f1 |= SECONDARY;
        f2 |= SECONDARY;
    }

    int template_len1 = 0;
    bool both_aligned = !alignment1.is_unaligned && !alignment2.is_unaligned;
    if (both_aligned && alignment1.ref_id == alignment2.ref_id) {
        const int dist = alignment2.ref_start - alignment1.ref_start;
        if (dist > 0) {
            template_len1 = dist + alignment2.length;
        }
        else {
            template_len1 = dist - alignment1.length;
        }
    }
    if (is_proper) {
        f1 |= PROPER_PAIR;
        f2 |= PROPER_PAIR;
    }

    std::string reference_name1;
    int pos1 = alignment1.ref_start;
    int edit_distance1 = alignment1.edit_distance;
    if (alignment1.is_unaligned) {
        f1 |= UNMAP;
        f2 |= MUNMAP;
        pos1 = -1;
        reference_name1 = "*";
    } else {
        if (alignment1.is_rc) {
            f1 |= REVERSE;
            f2 |= MREVERSE;
        }
        reference_name1 = references.names[alignment1.ref_id];
    }

    std::string reference_name2;
    int pos2 = alignment2.ref_start;
    int edit_distance2 = alignment2.edit_distance;
    if (alignment2.is_unaligned) {
        f2 |= UNMAP;
        f1 |= MUNMAP;
        pos2 = -1;
        reference_name2 = "*";
    } else {
        if (alignment2.is_rc) {
            f1 |= MREVERSE;
            f2 |= REVERSE;
        }
        reference_name2 = references.names[alignment2.ref_id];
    }

    // Reference name as used in the RNEXT field;
    // set to "=" if identical to reference_name
    std::string mate_reference_name1 = reference_name1;
    std::string mate_reference_name2 = reference_name2;
    if (
        (!alignment1.is_unaligned && !alignment2.is_unaligned && alignment1.ref_id == alignment2.ref_id)
        || (alignment1.is_unaligned != alignment2.is_unaligned)
    ) {
        mate_reference_name1 = "=";
        mate_reference_name2 = "=";
    }

    if (alignment1.is_unaligned != alignment2.is_unaligned) {
        if (alignment1.is_unaligned) {
            pos1 = pos2;
        } else {
            pos2 = pos1;
        }
    }

    if (alignment1.is_unaligned) {
        add_unmapped_mate(record1, f1, reference_name2, pos2);
    } else {
        add_record(record1.name, f1, reference_name1, alignment1.ref_start, mapq1, alignment1.cigar, mate_reference_name2, pos2, template_len1, record1.seq, read1_rc, record1.qual, edit_distance1, alignment1.score, details[0]);
    }
    if (alignment2.is_unaligned) {
        add_unmapped_mate(record2, f2, reference_name1, pos1);
    } else {
        add_record(record2.name, f2, reference_name2, alignment2.ref_start, mapq2, alignment2.cigar, mate_reference_name1, pos1, -template_len1, record2.seq, read2_rc, record2.qual, edit_distance2, alignment2.score, details[1]);
    }
}

bool is_proper_pair(const Alignment& alignment1, const Alignment& alignment2, float mu, float sigma) {
    const int dist = alignment2.ref_start - alignment1.ref_start;
    const bool same_reference = alignment1.ref_id == alignment2.ref_id;
    const bool both_aligned = same_reference && !alignment1.is_unaligned && !alignment2.is_unaligned;
    const bool r1_r2 = !alignment1.is_rc && alignment2.is_rc && dist >= 0; // r1 ---> <---- r2
    const bool r2_r1 = !alignment2.is_rc && alignment1.is_rc && dist <= 0; // r2 ---> <---- r1
    const bool rel_orientation_good = r1_r2 || r2_r1;
    const bool insert_good = std::abs(dist) <= mu + 6 * sigma;

    return both_aligned && insert_good && rel_orientation_good;
}
