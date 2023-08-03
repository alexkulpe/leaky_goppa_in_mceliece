#include <stdint.h>
#include <stdio.h>


/*
gets the lexicographically next integer after x with hammingweith hw(x)
if this is higher than limit, then returns 0

*/
uint64_t next_subset(uint64_t x, uint64_t limit)
{
    if (x == 0)
    {
        return 0;
    }
    uint64_t y = x & -x;
    uint64_t c = x + y;
    x = (((x ^ c) >> 2) / y) | c;
    if (x & limit)
    {
        x = ((x & (limit - 1)) << 2) | 3;
    }
    return x;
}
