"""
0, 1, 0, 4, 9, 4, 11, 18, 12, 26, 20, 23, 32, 45, 53, 46, 59, 66, 60, 70, 80, 84, 97, 107, 128, 122, 140, 113, 136, 159, 148, 141, 155, 184, 161, 174, 165, 171, 214
, 220, 216, 222, 219, 251, 270, 249, 252, 266, 277, 291, 285, 301, 282, 293, 300, 345, 334, 356, 351, 368, 334, 375, 369, 332, 363, 338, 344, 387, 370, 342, 338, 34
9, 344, 369, 374, 339, 347, 375, 393, 340, 380, 350, 356, 349, 364, 362, 357, 346, 323, 361, 375, 385, 378, 376, 350, 353, 383, 369, 372, 383, 345, 332, 401, 367, 3
71, 385, 341, 379, 361, 368, 355, 370, 382, 322, 380, 345, 345, 341, 350, 366, 358, 392, 328, 320, 387, 358, 375, 358, 365, 348, 375, 392, 352, 363, 324, 346, 363, 
331, 359, 405, 381, 371, 386, 356, 361, 373, 361, 345, 353, 358, 360, 356, 391, 339, 323, 374, 319, 396, 371, 374, 370, 381, 356, 358, 332, 343, 356, 349, 352, 361,
 359, 364, 318, 363, 353, 336, 361, 370, 350, 355, 373, 374, 353, 339, 332, 383, 350, 343, 372, 352, 337, 330, 325, 322, 326, 313, 324, 323, 304, 321, 290, 277, 276
, 262, 255, 248, 263, 251, 219, 241, 229, 197, 206, 192, 210, 190, 190, 180, 161, 155, 157, 162, 153, 161, 126, 110, 121, 115, 97, 87, 72, 77, 67, 61, 68, 38, 35, 3
9, 39, 26, 19, 30, 23, 15, 8, 8, 4, 4, 3, 2, 0, 1, 0, 0, 0, 0, check total sum: 65536                                                                               
check hist CDF                                                                                                                                                      
0, 1, 1, 5, 14, 18, 29, 47, 59, 85, 105, 128, 160, 205, 258, 304, 363, 429, 489, 559, 639, 723, 820, 927, 1055, 1177, 1317, 1430, 1566, 1725, 1873, 2014, 2169, 2353
, 2514, 2688, 2853, 3024, 3238, 3458, 3674, 3896, 4115, 4366, 4636, 4885, 5137, 5403, 5680, 5971, 6256, 6557, 6839, 7132, 7432, 7777, 8111, 8467, 8818, 9186, 9520, 
9895, 10264, 10596, 10959, 11297, 11641, 12028, 12398, 12740, 13078, 13427, 13771, 14140, 14514, 14853, 15200, 15575, 15968, 16308, 16688, 17038, 17394, 17743, 1810
7, 18469, 18826, 19172, 19495, 19856, 20231, 20616, 20994, 21370, 21720, 22073, 22456, 22825, 23197, 23580, 23925, 24257, 24658, 25025, 25396, 25781, 26122, 26501, 
26862, 27230, 27585, 27955, 28337, 28659, 29039, 29384, 29729, 30070, 30420, 30786, 31144, 31536, 31864, 32184, 32571, 32929, 33304, 33662, 34027, 34375, 34750, 351
42, 35494, 35857, 36181, 36527, 36890, 37221, 37580, 37985, 38366, 38737, 39123, 39479, 39840, 40213, 40574, 40919, 41272, 41630, 41990, 42346, 42737, 43076, 43399,
 43773, 44092, 44488, 44859, 45233, 45603, 45984, 46340, 46698, 47030, 47373, 47729, 48078, 48430, 48791, 49150, 49514, 49832, 50195, 50548, 50884, 51245, 51615, 51
965, 52320, 52693, 53067, 53420, 53759, 54091, 54474, 54824, 55167, 55539, 55891, 56228, 56558, 56883, 57205, 57531, 57844, 58168, 58491, 58795, 59116, 59406, 59683
, 59959, 60221, 60476, 60724, 60987, 61238, 61457, 61698, 61927, 62124, 62330, 62522, 62732, 62922, 63112, 63292, 63453, 63608, 63765, 63927, 64080, 64241, 64367, 6
4477, 64598, 64713, 64810, 64897, 64969, 65046, 65113, 65174, 65242, 65280, 65315, 65354, 65393, 65419, 65438, 65468, 65491, 65506, 65514, 65522, 65526, 65530, 6553
3, 65535, 65535, 65536, 65536, 65536, 65536, 65536, (0, 0, 0): 0.909804check solution image      
"""
a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 3, 8, 16, 18, 22, 34, 24, 46, 64, 97, 103, 125, 133, 186, 214, 193, 220, 246, 255, 276, 296, 322, 354, 343, 345, 348, 360, 378, 392, 427, 434, 476, 525, 515, 582, 581, 566, 589, 576, 589, 678, 761, 814, 898, 781, 755, 772, 884, 880, 853, 804, 796, 722, 656, 566, 571, 503, 548, 547, 523, 552, 505, 524, 533, 495, 548, 544, 568, 579, 542, 526, 553, 532, 449, 510, 562, 500, 455, 501, 477, 530, 591, 519, 558, 641, 749, 799, 700, 644, 617, 618, 
649, 683, 807, 682, 774, 933, 868, 1225, 1007, 1379, 1359, 1447, 2536, 1885, 537, 143, 164, 148, 147, 147, 142, 140, 151, 156, 130, 136, 129, 117, 106, 135, 130, 133, 125, 112, 135, 127, 114, 107, 109, 96, 108, 111, 106, 100, 111, 100, 100, 93, 88, 99, 99, 113, 100, 81, 82, 86, 73, 84, 74, 57, 58, 61, 54, 39, 41, 45, 35, 38, 29, 23, 32, 15, 16, 17, 13, 11, 7, 9, 7, 9, 6, 5, 5, 2, 3, 1, 2, 5, 4, 6, 2, 0, 0, 0]

b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 6, 7, 10, 18, 34, 52, 74, 108, 132, 178, 242, 339, 442, 567, 700, 886, 1100, 1293, 1513, 1759, 2014, 2290, 2586, 2908, 3262, 3605, 3950, 4298, 4658, 5036, 5428, 5855, 6289, 6765, 7290, 7805, 8387, 8968, 9534, 10123, 10699, 11288, 11966, 12727, 13541, 14439, 15220, 15975, 16747, 17631,
 18511, 19364, 20168, 20964, 21686, 22342, 22908, 23479, 23982, 24530, 25077, 25600, 26152, 26657, 27181, 27714, 28209, 28757, 29301, 29869, 30448, 30990, 31516, 32069, 32601, 33050, 33560, 34122, 34622, 35077, 35578, 36055, 36585, 37176, 37695, 38253, 38894, 39643, 40442, 41142, 41786, 42403, 43021, 43670, 44353, 45160, 45842
, 46616, 47549, 48417, 49642, 50649, 52028, 53387, 54834, 57370, 59255, 59792, 59935, 60099, 60247, 60394, 60541, 60683, 60823, 60974, 61130, 61260, 61396, 61525, 61642, 61748, 61883, 62013, 62146, 62271, 62383, 62518, 62645, 62759, 62866, 62975, 63071, 63179, 63290, 63396, 63496, 63607, 63707, 63807, 63900, 63988, 64087, 64186, 64299, 64399, 64480, 64562, 64648, 64721, 64805, 64879, 64936, 64994, 65055, 65109, 65148, 65189, 65234, 65269, 65307, 65336, 65359, 65391, 65406, 65422, 65439, 
65452, 65463, 65470, 65479, 65486, 65495, 65501, 65506, 65511, 65513, 65516, 65517, 65519, 65524, 65528, 65534, 65536, 65536, 65536, 65536]

res = 0
for index, item in enumerate(a):
    res += item

print(res)