# skip_utils.py
#
# Skip pointers in conjunctive queries.
# Using skipping as specified in Figure 2.10 in Manning et al.'s book.
# Modified for this projects index setup
#
# Author: R. Zanibbi

import math


def skipLength(plist):
    return math.floor(math.sqrt(len(plist)))


def skip_intersect(plist1, plist2):
    # Implementation of Figure 1.6 from Manning's book to intersect two
    # posting lists. Adding element for the bonus, to count comparisons.
    answer = []
    comparisonCount = 0

    # Find common documents (i.e., intersection of plist1 and plist2)
    # Compute skip pointer increment (as floor of root of posting list length)
    i = 0
    j = 0
    iskip = skipLength(plist1)
    jskip = skipLength(plist2)

    while i < len(plist1) and j < len(plist2):

        comparisonCount += 1
        val1 = plist1[i].split(":")[0]
        val2 = plist2[i].split(":")[0]
        if val1 == val2:
            answer += [val1]
            i += 1
            j += 1

        else:
            # NOTE: this else branch is where we apply skips; 'hasSkip()' always
            # true in our case, because the postings are not compressed.
            # Try to skip in first list.
            comparisonCount += 1
            val2 = plist2[j].split(":")[0]
            if val1 < val2:
                comparisonCount += 1
                if (i + iskip) < len(plist1) and plist1[i + iskip].split(":")[0] <= val2:
                    comparisonCount += 1
                    while (i + iskip) < len(plist1) and plist1[i + iskip].split(":")[0] <= val2:
                        i += iskip
                        comparisonCount += 1
                else:
                    i += 1

            # Try to skip in the second list.
            else:
                comparisonCount += 1
                if (j + jskip) < len(plist2) and plist2[j + jskip].split(":")[0] <= val1:
                    comparisonCount += 1
                    while (j + jskip) < len(plist2) and plist2[j + jskip].split(":")[0] <= val1:
                        comparisonCount += 1
                        j += jskip
                else:
                    j += 1

    return (answer, comparisonCount)