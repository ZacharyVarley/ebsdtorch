"""

Module for handling group operations. My space group closures (enumerating all
matrices for equivalent positions from a starting set of generators) are 1000
times slower (at least) than gemmi's implementation, but it's all in PyTorch.

10s of milliseconds is still fast enough for most purposes as this is usually
only done once per structure. If you are initializing 1000s of structures, you
may want to use gemmi's implementation or precompute all operators for all space
groups.

See the bottom of this page for more info:

https://gemmi.readthedocs.io/en/latest/symmetry.html

The code below is basically avoiding pasting a 10,000 line precomputed table of
operators:

https://github.com/openbabel/openbabel/blob/master/data/space-groups.txt

"""

import torch
from torch import Tensor


# fmt: off
# Space groups with two settings
sg_two_settings = [
    48, 50, 59, 68, 
    70, 85, 86, 88, 
    125, 126, 129, 130, 
    133, 134, 137, 138, 
    141, 142, 201, 203, 
    222, 224, 227, 228,
    ]
# Point group names
pg_names =[
    '    1','   -1','    2','    m','  2/m','  222', 
    '  mm2','  mmm','    4','   -4','  4/m','  422',
    '  4mm',' -42m','4/mmm','    3','   -3','   32',
    '   3m','  -3m','    6','   -6','  6/m','  622',
    '  6mm',' -6m2','6/mmm','   23','   m3','  432',
    ' -43m',' m-3m','  532','  822',' 1022',' 1222',
    ]

# Space group names
sg_names = (" P  1      " ," P -1      ", # MONOCLINIC SPACE GROUPS
        " P 2       " ," P 21      " ," C 2       " ," P m       ", 
        " P c       " ," C m       " ," C c       " ," P 2/m     ", 
        " P 21/m    " ," C 2/m     " ," P 2/c     " ," P 21/c    ", 
        " C 2/c     ",                                               # ORTHORHOMBIC SPACE GROUPS
        " P 2 2 2   " ," P 2 2 21  " ," P 21 21 2 " ," P 21 21 21", 
        " C 2 2 21  " ," C 2 2 2   " ," F 2 2 2   " ," I 2 2 2   ", 
        " I 21 21 21" ," P m m 2   " ," P m c 21  " ," P c c 2   ", 
        " P m a 2   " ," P c a 21  " ," P n c 2   " ," P m n 21  ", 
        " P b a 2   " ," P n a 21  " ," P n n 2   " ," C m m 2   ", 
        " C m c 21  " ," C c c 2   " ," A m m 2   " ," A b m 2   ", 
        " A m a 2   " ," A b a 2   " ," F m m 2   " ," F d d 2   ", 
        " I m m 2   " ," I b a 2   " ," I m a 2   " ," P m m m   ", 
        " P n n n   " ," P c c m   " ," P b a n   " ," P m m a   ", 
        " P n n a   " ," P m n a   " ," P c c a   " ," P b a m   ", 
        " P c c n   " ," P b c m   " ," P n n m   " ," P m m n   ", 
        " P b c n   " ," P b c a   " ," P n m a   " ," C m c m   ", 
        " C m c a   " ," C m m m   " ," C c c m   " ," C m m a   ", 
        " C c c a   " ," F m m m   " ," F d d d   " ," I m m m   ", 
        " I b a m   " ," I b c a   " ," I m m a   ",                 # TETRAGONAL SPACE GROUPS  
        " P 4       " ," P 41      " ," P 42      " ," P 43      ", 
        " I 4       " ," I 41      " ," P -4      " ," I -4      ", 
        " P 4/m     " ," P 42/m    " ," P 4/n     " ," P 42/n    ", 
        " I 4/m     " ," I 41/a    " ," P 4 2 2   " ," P 4 21 2  ", 
        " P 41 2 2  " ," P 41 21 2 " ," P 42 2 2  " ," P 42 21 2 ", 
        " P 43 2 2  " ," P 43 21 2 " ," I 4 2 2   " ," I 41 2 2  ", 
        " P 4 m m   " ," P 4 b m   " ," P 42 c m  " ," P 42 n m  ", 
        " P 4 c c   " ," P 4 n c   " ," P 42 m c  " ," P 42 b c  ", 
        " I 4 m m   " ," I 4 c m   " ," I 41 m d  " ," I 41 c d  ", 
        " P -4 2 m  " ," P -4 2 c  " ," P -4 21 m " ," P -4 21 c ", 
        " P -4 m 2  " ," P -4 c 2  " ," P -4 b 2  " ," P -4 n 2  ", 
        " I -4 m 2  " ," I -4 c 2  " ," I -4 2 m  " ," I -4 2 d  ", 
        " P 4/m m m " ," P 4/m c c " ," P 4/n b m " ," P 4/n n c ", 
        " P 4/m b m " ," P 4/m n c " ," P 4/n m m " ," P 4/n c c ", 
        " P 42/m m c" ," P 42/m c m" ," P 42/n b c" ," P 42/n n m", 
        " P 42/m b c" ," P 42/m n m" ," P 42/n m c" ," P 42/n c m", 
        " I 4/m m m " ," I 4/m c m " ," I 41/a m d" ," I 41/a c d",  # RHOMBOHEDRAL SPACE GROUPS  
        " P 3       " ," P 31      " ," P 32      " ," R 3       ", 
        " P -3      " ," R -3      " ," P 3 1 2   " ," P 3 2 1   ", 
        " P 31 1 2  " ," P 31 2 1  " ," P 32 1 2  " ," P 32 2 1  ", 
        " R 3 2     " ," P 3 m 1   " ," P 3 1 m   " ," P 3 c 1   ", 
        " P 3 1 c   " ," R 3 m     " ," R 3 c     " ," P -3 1 m  ", 
        " P -3 1 c  " ," P -3 m 1  " ," P -3 c 1  " ," R -3 m    ", 
        " R -3 c    ",                                               # HEXAGONAL SPACE GROUPS   
        " P 6       " ," P 61      " ," P 65      " ," P 62      ", 
        " P 64      " ," P 63      " ," P -6      " ," P 6/m     ", 
        " P 63/m    " ," P 6 2 2   " ," P 61 2 2  " ," P 65 2 2  ", 
        " P 62 2 2  " ," P 64 2 2  " ," P 63 2 2  " ," P 6 m m   ", 
        " P 6 c c   " ," P 63 c m  " ," P 63 m c  " ," P -6 m 2  ", 
        " P -6 c 2  " ," P -6 2 m  " ," P -6 2 c  " ," P 6/m m m ", 
        " P 6/m c c " ," P 63/m c m" ," P 63/m m c",                 # CUBIC SPACE GROUPS
        " P 2 3     " ," F 2 3     " ," I 2 3     " ," P 21 3    ", 
        " I 21 3    " ," P m 3     " ," P n 3     " ," F m 3     ", 
        " F d 3     " ," I m 3     " ," P a 3     " ," I a 3     ", 
        " P 4 3 2   " ," P 42 3 2  " ," F 4 3 2   " ," F 41 3 2  ", 
        " I 4 3 2   " ," P 43 3 2  " ," P 41 3 2  " ," I 41 3 2  ", 
        " P -4 3 m  " ," F -4 3 m  " ," I -4 3 m  " ," P -4 3 n  ", 
        " F -4 3 c  " ," I -4 3 d  " ," P m 3 m   " ," P n 3 n   ", 
        " P m 3 n   " ," P n 3 m   " ," F m 3 m   " ," F m 3 c   ", 
        " F d 3 m   " ," F d 3 c   " ," I m 3 m   " ," I a 3 d   ",  # TRIGONAL GROUPS RHOMBOHEDRAL SETTING
        " R 3   |146" ," R -3  |148" ," R 3 2 |155" ," R 3 m |160", 
        " R 3 c |161" ," R -3 m|166" ," R -3 c|167")


sg_gen_strings = [
    "000                                     ", "100                                     ", "01cOOO0                                 ",
    "01cODO0                                 ", "02aDDOcOOO0                             ", "01jOOO0                                 ",
    "01jOOD0                                 ", "02aDDOjOOO0                             ", "02aDDOjOOD0                             ",
    "11cOOO0                                 ", "11cODO0                                 ", "12aDDOcOOO0                             ",
    "11cOOD0                                 ", "11cODD0                                 ", "12aDDOcOOD0                             ",
    "02bOOOcOOO0                             ", "02bOODcOOD0                             ", "02bOOOcDDO0                             ",
    "02bDODcODD0                             ", "03aDDObOODcOOD0                         ", "03aDDObOOOcOOO0                         ",
    "04aODDaDODbOOOcOOO0                     ", "03aDDDbOOOcOOO0                         ", "03aDDDbDODcODD0                         ",
    "02bOOOjOOO0                             ", "02bOODjOOD0                             ", "02bOOOjOOD0                             ",
    "02bOOOjDOO0                             ", "02bOODjDOO0                             ", "02bOOOjODD0                             ",
    "02bDODjDOD0                             ", "02bOOOjDDO0                             ", "02bOODjDDO0                             ",
    "02bOOOjDDD0                             ", "03aDDObOOOjOOO0                         ", "03aDDObOODjOOD0                         ",
    "03aDDObOOOjOOD0                         ", "03aODDbOOOjOOO0                         ", "03aODDbOOOjODO0                         ",
    "03aODDbOOOjDOO0                         ", "03aODDbOOOjDDO0                         ", "04aODDaDODbOOOjOOO0                     ",
    "04aODDaDODbOOOjBBB0                     ", "03aDDDbOOOjOOO0                         ", "03aDDDbOOOjDDO0                         ",
    "03aDDDbOOOjDOO0                         ", "12bOOOcOOO0                             ", "03bOOOcOOOhDDD1BBB                      ",
    "12bOOOcOOD0                             ", "03bOOOcOOOhDDO1BBO                      ", "12bDOOcOOO0                             ",
    "12bDOOcDDD0                             ", "12bDODcDOD0                             ", "12bDOOcOOD0                             ",
    "12bOOOcDDO0                             ", "12bDDOcODD0                             ", "12bOODcODD0                             ",
    "12bOOOcDDD0                             ", "03bOOOcDDOhDDO1BBO                      ", "12bDDDcOOD0                             ",
    "12bDODcODD0                             ", "12bDODcODO0                             ", "13aDDObOODcOOD0                         ",
    "13aDDObODDcODD0                         ", "13aDDObOOOcOOO0                         ", "13aDDObOOOcOOD0                         ",
    "13aDDObODOcODO0                         ", "04aDDObDDOcOOOhODD1OBB                  ", "14aODDaDODbOOOcOOO0                     ",
    "05aODDaDODbOOOcOOOhBBB1ZZZ              ", "13aDDDbOOOcOOO0                         ", "13aDDDbOOOcDDO0                         ",
    "13aDDDbDODcODD0                         ", "13aDDDbODOcODO0                         ", "02bOOOgOOO0                             ",
    "02bOODgOOB0                             ", "02bOOOgOOD0                             ", "02bOODgOOF0                             ",
    "03aDDDbOOOgOOO0                         ", "03aDDDbDDDgODB0                         ", "02bOOOmOOO0                             ",
    "03aDDDbOOOmOOO0                         ", "12bOOOgOOO0                             ", "12bOOOgOOD0                             ",
    "03bOOOgDDOhDDO1YBO                      ", "03bOOOgDDDhDDD1YYY                      ", "13aDDDbOOOgOOO0                         ",
    "04aDDDbDDDgODBhODB1OYZ                  ", "03bOOOgOOOcOOO0                         ", "03bOOOgDDOcDDO0                         ",
    "03bOODgOOBcOOO0                         ", "03bOODgDDBcDDB0                         ", "03bOOOgOODcOOO0                         ",
    "03bOOOgDDDcDDD0                         ", "03bOODgOOFcOOO0                         ", "03bOODgDDFcDDF0                         ",
    "04aDDDbOOOgOOOcOOO0                     ", "04aDDDbDDDgODBcDOF0                     ", "03bOOOgOOOjOOO0                         ",
    "03bOOOgOOOjDDO0                         ", "03bOOOgOODjOOD0                         ", "03bOOOgDDDjDDD0                         ",
    "03bOOOgOOOjOOD0                         ", "03bOOOgOOOjDDD0                         ", "03bOOOgOODjOOO0                         ",
    "03bOOOgOODjDDO0                         ", "04aDDDbOOOgOOOjOOO0                     ", "04aDDDbOOOgOOOjOOD0                     ",
    "04aDDDbDDDgODBjOOO0                     ", "04aDDDbDDDgODBjOOD0                     ", "03bOOOmOOOcOOO0                         ",
    "03bOOOmOOOcOOD0                         ", "03bOOOmOOOcDDO0                         ", "03bOOOmOOOcDDD0                         ",
    "03bOOOmOOOjOOO0                         ", "03bOOOmOOOjOOD0                         ", "03bOOOmOOOjDDO0                         ",
    "03bOOOmOOOjDDD0                         ", "04aDDDbOOOmOOOjOOO0                     ", "04aDDDbOOOmOOOjOOD0                     ",
    "04aDDDbOOOmOOOcOOO0                     ", "04aDDDbOOOmOOOcDOF0                     ", "13bOOOgOOOcOOO0                         ",
    "13bOOOgOOOcOOD0                         ", "04bOOOgOOOcOOOhDDO1YYO                  ", "04bOOOgOOOcOOOhDDD1YYY                  ",
    "13bOOOgOOOcDDO0                         ", "13bOOOgOOOcDDD0                         ", "04bOOOgDDOcDDOhDDO1YBO                  ",
    "04bOOOgDDOcDDDhDDO1YBO                  ", "13bOOOgOODcOOO0                         ", "13bOOOgOODcOOD0                         ",
    "04bOOOgDDDcOODhDDD1YBY                  ", "04bOOOgDDDcOOOhDDD1YBY                  ", "13bOOOgOODcDDO0                         ",
    "13bOOOgDDDcDDD0                         ", "04bOOOgDDDcDDDhDDD1YBY                  ", "04bOOOgDDDcDDOhDDD1YBY                  ",
    "14aDDDbOOOgOOOcOOO0                     ", "14aDDDbOOOgOOOcOOD0                     ", "05aDDDbDDDgODBcDOFhODB1OBZ              ",
    "05aDDDbDDDgODBcDOBhODB1OBZ              ", "01nOOO0                                 ", "01nOOC0                                 ",
    "01nOOE0                                 ", "02aECCnOOO0                             ", "11nOOO0                                 ",
    "12aECCnOOO0                             ", "02nOOOfOOO0                             ", "02nOOOeOOO0                             ",
    "02nOOCfOOE0                             ", "02nOOCeOOO0                             ", "02nOOEfOOC0                             ",
    "02nOOEeOOO0                             ", "03aECCnOOOeOOO0                         ", "02nOOOkOOO0                             ",
    "02nOOOlOOO0                             ", "02nOOOkOOD0                             ", "02nOOOlOOD0                             ",
    "03aECCnOOOkOOO0                         ", "03aECCnOOOkOOD0                         ", "12nOOOfOOO0                             ",
    "12nOOOfOOD0                             ", "12nOOOeOOO0                             ", "12nOOOeOOD0                             ",
    "13aECCnOOOeOOO0                         ", "13aECCnOOOeOOD0                         ", "02nOOObOOO0                             ",
    "02nOOCbOOD0                             ", "02nOOEbOOD0                             ", "02nOOEbOOO0                             ",
    "02nOOCbOOO0                             ", "02nOOObOOD0                             ", "02nOOOiOOO0                             ",
    "12nOOObOOO0                             ", "12nOOObOOD0                             ", "03nOOObOOOeOOO0                         ",
    "03nOOCbOODeOOC0                         ", "03nOOEbOODeOOE0                         ", "03nOOEbOOOeOOE0                         ",
    "03nOOCbOOOeOOC0                         ", "03nOOObOODeOOO0                         ", "03nOOObOOOkOOO0                         ",
    "03nOOObOOOkOOD0                         ", "03nOOObOODkOOD0                         ", "03nOOObOODkOOO0                         ",
    "03nOOOiOOOkOOO0                         ", "03nOOOiOODkOOD0                         ", "03nOOOiOOOeOOO0                         ",
    "03nOOOiOODeOOO0                         ", "13nOOObOOOeOOO0                         ", "13nOOObOOOeOOD0                         ",
    "13nOOObOODeOOD0                         ", "13nOOObOODeOOO0                         ", "03bOOOcOOOdOOO0                         ",
    "05aODDaDODbOOOcOOOdOOO0                 ", "04aDDDbOOOcOOOdOOO0                     ", "03bDODcODDdOOO0                         ",
    "04aDDDbDODcODDdOOO0                     ", "13bOOOcOOOdOOO0                         ", "04bOOOcOOOdOOOhDDD1YYY                  ",
    "15aODDaDODbOOOcOOOdOOO0                 ", "06aODDaDODbOOOcOOOdOOOhBBB1ZZZ          ", "14aDDDbOOOcOOOdOOO0                     ",
    "13bDODcODDdOOO0                         ", "14aDDDbDODcODDdOOO0                     ", "04bOOOcOOOdOOOeOOO0                     ",
    "04bOOOcOOOdOOOeDDD0                     ", "06aODDaDODbOOOcOOOdOOOeOOO0             ", "06aODDaDODbODDcDDOdOOOeFBF0             ",
    "05aDDDbOOOcOOOdOOOeOOO0                 ", "04bDODcODDdOOOeBFF0                     ", "04bDODcODDdOOOeFBB0                     ",
    "05aDDDbDODcODDdOOOeFBB0                 ", "04bOOOcOOOdOOOlOOO0                     ", "06aODDaDODbOOOcOOOdOOOlOOO0             ",
    "05aDDDbOOOcOOOdOOOlOOO0                 ", "04bOOOcOOOdOOOlDDD0                     ", "06aODDaDODbOOOcOOOdOOOlDDD0             ",
    "05aDDDbDODcODDdOOOlBBB0                 ", "14bOOOcOOOdOOOeOOO0                     ", "05bOOOcOOOdOOOeOOOhDDD1YYY              ",
    "14bOOOcOOOdOOOeDDD0                     ", "05bOOOcOOOdOOOeDDDhDDD1YYY              ", "16aODDaDODbOOOcOOOdOOOeOOO0             ",
    "16aODDaDODbOOOcOOOdOOOeDDD0             ", "07aODDaDODbODDcDDOdOOOeFBFhBBB1ZZZ      ", "07aODDaDODbODDcDDOdOOOeFBFhFFF1XXX      ",
    "15aDDDbOOOcOOOdOOOeOOO0                 ", "15aDDDbDODcODDdOOOeFBB0                 ", "01dOOO0                                 ",
    "11dOOO0                                 ", "02dOOOfOOO0                             ", "02dOOOlOOO0                             ",
    "02dOOOlDDD0                             ", "12dOOOfOOO0                             ", "12dOOOfDDD0                             ",
]

sg_ords = [1,2,2,2,4,2,2,4,4,4,4,8,4,4,8,4,4,4,4,8,8,16,8,8,4,4,4,4,4,4,4,4,4,4,8,8,8,8,
           8,8,8,16,16,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,16,16,16,16,16,16,32,32,16,
           16,16,16,4,4,4,4,8,8,4,8,8,8,8,8,16,16,8,8,8,8,8,8,8,8,16,16,8,8,8,8,8,8,8,8,
           16,16,16,16,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
           16,16,16,32,32,32,32,3,3,3,9,6,18,6,6,6,6,6,6,18,6,6,6,6,18,18,12,12,12,12,36,
           36,6,6,6,6,6,6,6,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,24,24,24,24,
           12,48,24,12,24,24,24,96,96,48,24,48,24,24,96,96,48,24,24,48,24,96,48,24,96,48,
           48,48,48,48,192,192,192,192,96,96,]

sg_extended_orth_settings = [
    " a  b  c", " b  a -c", " c  a  b", "-c  b  a", " b  c  a", " a -c  b"
]

sg_xt_beg = [1, 3, 16, 75, 143, 168, 195]
sg_pg_beg = [
    1,2,3,6,10,16,25,47,75,81,83,89,99,111,123,143,
    147,149,156,162,168,174,175,177,183,187,191,195,
    200,207,215,221
    ]
sg_symmorphic = [
    1,2,3,5,6,8,10,12,16,21,22,23,25,35,38,42,44,47,
    65,69,71,75,79,81,82,83,87,89,97,99,107,111,115,
    119,121,123,139,143,146,147,148,149,150,155,156,
    157,160,162,164,166,168,174,175,177,183,187,189,
    191,195,196,197,200,202,204,207,209,211,215,216,
    217,221,225,229,
    ]

pg_to_laue = [1,1,2,2,2,3,3,3,
              4,4,4,5,5,5,5,6,
              6,7,7,7,8,8,8,9,
              9,9,9,10,10,11,11,11,
              12,13,14,15, # 
              ]

# fmt: on


def read_sg_gen_string(
    sg: int,
    sg_setting: int = 1,
    sg_gen_strings: list = sg_gen_strings,
    eps: float = 5e-4,
) -> Tensor:
    """
    Read the generator string for a given space group.

    Args:
        sg: Space group number

    Returns:
        Generators for the space group

    """

    # Create mapping for translation components
    trans_map = {
        "A": 1.0 / 6.0,
        "B": 1.0 / 4.0,
        "C": 1.0 / 3.0,
        "D": 1.0 / 2.0,
        "E": 2.0 / 3.0,
        "F": 3.0 / 4.0,
        "G": 5.0 / 6.0,
        "O": 0.0,
        "X": -3.0 / 8.0,
        "Y": -1.0 / 4.0,
        "Z": -1.0 / 8.0,
    }

    # Create mapping for rotation matrices (first character)
    rot_matrices = {
        "a": torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64),
        "b": torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float64),
        "c": torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float64),
        "d": torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64),
        "e": torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=torch.float64),
        "f": torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=torch.float64),
        "g": torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64),
        "h": torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float64),
        "i": torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float64),
        "j": torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float64),
        "k": torch.tensor([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float64),
        "l": torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64),
        "m": torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=torch.float64),
        "n": torch.tensor([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=torch.float64),
    }

    # Initialize generator string (0 indexing instead of 1)
    gen_string = sg_gen_strings[sg - 1]
    gen_num = int(gen_string[1])

    identity = torch.eye(4, dtype=torch.float64)

    # start with identity operator
    gens = [identity]

    # potential inversion symmetry
    if gen_string[0] == "1":
        inversion = -1.0 * identity.clone()
        inversion[-1, -1] = 1.0
        gens.append(inversion)

    # loop
    for i in range(2, 2 + gen_num):
        t = gen_string[4 * (i - 2) + 2 : 4 * (i - 2) + 6]
        # print(f"t number {i}: {t}")
        rot = identity.clone()
        rot[:3, :3] = rot_matrices[t[0]]
        rot[0, 3] = trans_map[t[1]]
        rot[1, 3] = trans_map[t[2]]
        rot[2, 3] = trans_map[t[3]]
        gens.append(rot)

    gens = torch.stack(gens)

    # now check for special origin conditions (choices 1 and 2)
    if (gen_string[2 + 4 * gen_num] != "0") and sg_setting == 2:
        print("Special origin condition")
        t = gen_string[2 + 4 * gen_num : (2 + 4 * gen_num) + 3]
        translate_forward = torch.tensor(
            [
                [1.0, 0.0, 0.0, -trans_map[t[0]]],
                [0.0, 1.0, 0.0, -trans_map[t[1]]],
                [0.0, 0.0, 1.0, -trans_map[t[2]]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        translate_backward = torch.tensor(
            [
                [1.0, 0.0, 0.0, trans_map[t[0]]],
                [0.0, 1.0, 0.0, trans_map[t[1]]],
                [0.0, 0.0, 1.0, trans_map[t[2]]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        gens = torch.matmul(
            translate_backward,
            torch.matmul(gens, translate_forward),
        )

        gens[:, :, 3] = torch.where(
            torch.abs(gens[:, :, 3]) < eps,
            torch.zeros_like(gens[:, :, 3]),
            gens[:, :, 3],
        )

        gens[:, :, 3] = torch.fmod(1000 + gens[:, :, 3], 1.0)

        gens[:, :, 3] = torch.where(
            torch.abs(gens[:, :, 3] - 1.0) < eps,
            torch.zeros_like(gens[:, :, 3]),
            gens[:, :, 3],
        )

    return gens


@torch.jit.script
def matmul_trans_mod(
    A: Tensor,
    B: Tensor,
    eps: float = 5e-4,
) -> Tensor:
    """
    Matrix multiplication with translation modulo operation.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Result of the matrix multiplication

    """

    # initialize result
    C = torch.matmul(A, B)
    # print(f"C shape: {C.shape} = A shape: {A.shape} x B shape: {B.shape}")

    # check for translation components
    C[..., :3, 3] = torch.where(
        torch.abs(C[..., :3, 3]) < eps,
        torch.zeros_like(C[..., :3, 3]),
        C[..., :3, 3],
    )

    # # just use fmod
    C[..., :3, 3] = torch.fmod(1000 + C[..., :3, 3], 1.0)

    C[..., :3, 3] = torch.where(
        torch.abs(C[..., :3, 3] - 1.0) < eps,
        torch.zeros_like(C[..., :3, 3]),
        C[..., :3, 3],
    )

    return C


@torch.jit.script
def unique_rows(a: Tensor, eps: float = 5e-4) -> Tensor:
    remove = torch.zeros_like(a[:, 0], dtype=torch.bool)
    for i in range(a.shape[0]):
        if not remove[i]:
            equals = torch.all(torch.abs(a[i, :] - a[(i + 1) :]) < eps, dim=1)
            remove[(i + 1) :] = torch.logical_or(remove[(i + 1) :], equals)
    return a[~remove]


def sg_operators(
    sg: int,
    sg_setting: int = 1,
    sg_gen_strings: list = sg_gen_strings,
    eps: float = 5e-4,
) -> Tensor:
    """
    Close the generators of a space group to get all operators.

    Args:
        sg: Space group number
        tol: Tolerance for closing

    Returns:
        Closed generators

    """

    gens = read_sg_gen_string(sg, sg_setting, sg_gen_strings, eps)
    target_ord = sg_ords[sg - 1]

    # depth is not more than 2 for most space groups
    for depth in range(
        3 if sg in [69, 202, 203, 204, 206, 225, 226, 227, 228, 229, 230] else 2
    ):
        # 1st (depth = 0), 2nd (depth = 1), but not 3rd depth (depth = 2)
        if depth < 10:
            # multiply all pairs of unique generators
            candidates = matmul_trans_mod(gens[:, None, ...], gens[None, ...], eps)
            candidates = candidates.view(-1, 16)
            unique_candidates = unique_rows(candidates, eps).view(-1, 4, 4)

            # concat unique generators if they're not already in gens
            new_mask = ~torch.any(
                torch.all(
                    (unique_candidates[:, None, :3, :] - gens[None, :, :3, :]).abs()
                    < eps,
                    dim=(2, 3),
                ),
                dim=1,
            )
            new_gens = unique_candidates[new_mask]

            if ~new_mask.any():
                break
            else:
                gens = torch.cat((gens, new_gens), dim=0)

        else:
            # loop per generator because only ~20 new are left to find
            # in these specific space groups with depth 3 (depth = 2)
            new_gens = torch.zeros(0, 4, 4, dtype=torch.float64)
            for i in range(1, len(gens)):
                candidates = matmul_trans_mod(gens[[i]], gens, eps)
                unique_candidates = unique_rows(candidates.view(-1, 16), eps).view(
                    -1, 4, 4
                )

                # concat unique generators if they're not already in gens
                new_mask = ~torch.any(
                    torch.all(
                        (unique_candidates[:, None, :3, :] - gens[None, :, :3, :]).abs()
                        < eps,
                        dim=(2, 3),
                    ),
                    dim=1,
                )
                gens = torch.cat((gens, unique_candidates[new_mask]), dim=0)
                if len(gens) >= target_ord:
                    break

    gens[:, :, 3] = torch.fmod(100 + gens[:, :, 3], 1.0)

    return gens


# # # test it out and print the order of each of the 230 space groups
# import gemmi
# import numpy as np

# if __name__ == "__main__":
#     for sg in range(1, 231):
#         ord_gemmi = len(gemmi.SpaceGroup(sg).operations())
#         ord_close = len(sg_operators(sg))
#         print(
#             f"Space group {sg:03d}: {sg_names[sg - 1]} has order {ord_gemmi} vs. {ord_close}"
#         )
#         if ord_gemmi != ord_close:
#             raise ValueError(f"Space group {sg:03d} has different orders")

#     # print ops for 205
#     ops = sg_operators(205)
#     print(ops)
#     print("-" * 80)
#     ops_gemmi = list(gemmi.SpaceGroup(205).operations())
#     print(ops_gemmi)

# # n_runs = 10
# # sg = 225
# # import time

# # start = time.time()
# # for i in range(n_runs):
# #     sg_operators(sg)
# # end = time.time()

# # print(f"Average time over {n_runs} runs: {(end - start) / n_runs:.10f} s")
