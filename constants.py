TYPE_AA, TYPE_CG = 1, 2
S_RATIO = 2.0

SIG, EPS = 2.338, 0.4157
R_CUT_AA, R_CUT_CG = 5.0, 10.0  # physical cutoffs (Å): fine / coarse
SCALE_A_CG, SCALE_B_CG = 8.0, 0.5

# neighbor list radius = max physical cutoff over all species
NBR_CUT = max(R_CUT_AA, R_CUT_CG)
