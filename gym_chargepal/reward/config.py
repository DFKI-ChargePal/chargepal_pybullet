""" Configuration file for the reward classes. """


NORMALIZED_DIST_SPEED_REWARD = {
    'w_dist': 10.0,
    'dst_exp': 0.4,
    'w_speed': 0.01,
    'lower_s_bound': 0.001,
    'lower_d_bound': 0.1,
}


NORMALIZED_DIST_REWARD = {
    'w_dist': 1.0,
    'dst_exp': 0.4,
    'final_dst_exp': 0.1,
}

NORMALIZED_DIST_TIME_REWARD = {
    'w_dist': 1.0,
    'dst_exp': 0.4,
}
