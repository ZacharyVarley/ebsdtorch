import torch


@torch.jit.script
def space_group_to_laue(space_group: int) -> int:
    if space_group < 1 or space_group > 230:
        raise ValueError("The space group must be between 1 and 230 inclusive.")
    if space_group > 206:
        laue_group = 11
    elif space_group > 193:
        laue_group = 10
    elif space_group > 176:
        laue_group = 9
    elif space_group > 167:
        laue_group = 8
    elif space_group > 155:
        laue_group = 7
    elif space_group > 142:
        laue_group = 6
    elif space_group > 88:
        laue_group = 5
    elif space_group > 74:
        laue_group = 4
    elif space_group > 15:
        laue_group = 3
    elif space_group > 2:
        laue_group = 2
    else:
        laue_group = 1
    return laue_group


@torch.jit.script
def point_group_to_laue(point_group: str) -> int:
    if point_group in ["m3-m", "4-3m", "432"]:
        laue_group = 11
    elif point_group in ["m3-", "23"]:
        laue_group = 10
    elif point_group in ["6/mmm", "6-m2", "6mm", "622"]:
        laue_group = 9
    elif point_group in ["6/m", "6-", "6"]:
        laue_group = 8
    elif point_group in ["3-m", "3m", "32"]:
        laue_group = 7
    elif point_group in ["3-", "3"]:
        laue_group = 6
    elif point_group in ["4/mmm", "4-2m", "4mm", "422"]:
        laue_group = 5
    elif point_group in ["4/m", "4-", "4"]:
        laue_group = 4
    elif point_group in ["mmm", "mm2", "222"]:
        laue_group = 3
    elif point_group in ["2/m", "m", "2"]:
        laue_group = 2
    elif point_group in ["1-", "1"]:
        laue_group = 1
    else:
        raise ValueError(
            f"The point group symbol is not recognized, as one of, "
            + "m3-m, 4-3m, 432, m3-, 23, 6/mmm, 6-m2, 6mm, 622, 6/m, "
            + "6-, 6, 3-m, 3m, 32, 3-, 3, 4/mmm, 4-2m, 4mm, 422, 4/m, "
            + "4-, 4, mmm, mm2, 222, 2/m, m, 2, 1-, 1"
        )
    return laue_group
