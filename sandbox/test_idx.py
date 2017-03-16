import numpy as np

x = np.arange(21)

part_out_of = (1, 3)
this_part = part_out_of[0]
all_parts = part_out_of[1]
part_lenght = int(len(x) / all_parts)

if this_part == all_parts:
    x1 = x[part_lenght * (this_part - 1):]
else:
    x1 = x[part_lenght * (this_part - 1): part_lenght * this_part]

print x1

part_out_of = (2, 3)
this_part = part_out_of[0]
all_parts = part_out_of[1]
part_lenght = int(len(x) / all_parts)

if this_part == all_parts:
    x1 = x[part_lenght * (this_part - 1):]
else:
    x1 = x[part_lenght * (this_part - 1): part_lenght * this_part]

print x1

part_out_of = (3, 3)
this_part = part_out_of[0]
all_parts = part_out_of[1]
part_lenght = int(len(x) / all_parts)

if this_part == all_parts:
    x1 = x[part_lenght * (this_part - 1):]
else:
    x1 = x[part_lenght * (this_part - 1): part_lenght * this_part]

print x1

part_out_of = (4, 3)
this_part = part_out_of[0]
all_parts = part_out_of[1]
part_lenght = int(len(x) / all_parts)

if this_part == all_parts:
    x1 = x[part_lenght * (this_part - 1):]
else:
    x1 = x[part_lenght * (this_part - 1): part_lenght * this_part]

print x1
