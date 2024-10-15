import numpy as np
import matplotlib.pyplot as plt

import helper_functions.biot_savart as biot_savart

# Generate initial discretisation paramters
def disc_generate(disc_init: list, disc_mult: list, disc_order: list) -> dict:     
       disc_dict = dict()

       # Values in order of: current, next, multiplier, is converged
       for i_param in range(3):
              disc_dict.update({disc_order[i_param]: {'vals' : np.array([disc_init[i_param], disc_init[i_param]*disc_mult[i_param]]),
                                                                          'mult': disc_mult[i_param],
                                                                          'conv': False}})
       
       disc_dict['N']['vals'] = (np.floor(disc_dict['N']['vals']/2)*2).astype(int)
       return disc_dict

# Update the discretisation parameters if case is not converged     
def disc_update(disc_dict: dict, disc_param: str) -> dict:
       disc_dict_new = disc_dict

       disc_dict_new[disc_param]['vals'] = np.array([disc_dict[disc_param]['vals'][1], disc_dict[disc_param]['vals'][1]*disc_dict[disc_param]['mult']])
       if disc_param == 'N':
              disc_dict['N']['vals'] = (np.floor(disc_dict['N']['vals']/2)*2).astype(int)
       return disc_dict_new

# Returns the current parameter to change
def curr_param(disc_order: list, i_case: int) -> str:
       if i_case == 0:
              return None
       else:
              return disc_order[(i_case-1) % 3]

# Return the parameter to increase for the next case
def param_to_increase(disc_dict: dict, disc_order: list, i_case: int, is_final: bool) -> dict:
       params = dict()

       for i, i_param in enumerate(disc_order):
              if i_case != 0 and (i_case-1) % 3 == i and not is_final:
                     params.update({i_param: disc_dict[i_param]['vals'][1]})
              else:
                     params.update({i_param: disc_dict[i_param]['vals'][0]})

       return params

# Determine if two cases are converged
# Returns True if converged, False if not
def is_converged(case_data_prev, case_data_curr, pnts: np.ndarray, final_compare: bool, nodes = [-1]) -> bool:
       # Return false if only one case run
       if case_data_prev == None:
              if not final_compare: return False
       
       # Compare beam Z displacements
       beam_pos_prev = _extract_beam_pos(case_data_prev)
       beam_pos_curr = _extract_beam_pos(case_data_curr)
       if not _converge_beam_disp(beam_pos_prev, beam_pos_curr, nodes, final_compare):
              if not final_compare: return False
       
       # pnt_vel_prev = biot_savart.biot_savart_points(case_data_prev, pnts)
       # pnt_vel_curr = biot_savart.biot_savart_points(case_data_curr, pnts)

       # # Plot induced Z velocity at all points
       # [_, ax] = plt.subplots(1, 1)
       # t_prev = [t/case_data_prev.ts for t in range(1, case_data_prev.ts + 1)]
       # t_curr = [t/case_data_curr.ts for t in range(1, case_data_curr.ts + 1)]
       # for i_pnt in range(pnts.shape[1]):
       #        ax.plot(t_prev, pnt_vel_prev[:, i_pnt, 2])
       #        ax.plot(t_curr, pnt_vel_curr[:, i_pnt, 2])
       # plt.xlabel("Relative Time")
       # plt.ylabel("u_{z} (m/s)")
       # plt.show()

       # if not _converge_vel_pnt(pnt_vel_prev, pnt_vel_curr, final_compare):
       #        if not final_compare: return False

       return True

# Determine if the beam vertical displacement has converged
# Tolerences to be set to reasonable values      
def _converge_beam_disp(beam_pos_prev: np.ndarray, beam_pos_curr: np.ndarray, nodes: list, final_compare: bool)-> bool:
       n_tstep = np.min([beam_pos_prev.shape[0], beam_pos_curr.shape[0]])
       n_node = np.min([beam_pos_prev.shape[1], beam_pos_curr.shape[1]])

       beam_pos_prev = _interp_beam_t_n(beam_pos_prev, n_tstep, n_node)
       beam_pos_curr = _interp_beam_t_n(beam_pos_curr, n_tstep, n_node)
        
       for i_node in range(len(nodes)):
              abs_err = np.abs(beam_pos_curr[:, nodes[i_node]]-beam_pos_prev[:, nodes[i_node]])
              rel_err = np.abs(np.ones_like(beam_pos_curr[:, nodes[i_node]]) - \
                            beam_pos_curr[:, nodes[i_node]]/beam_pos_prev[:, nodes[i_node]])
              
              abs_err_max = np.max(abs_err)
              abs_err_avg = np.sum(abs_err)/n_tstep
              rel_err_max = np.max(rel_err)
              rel_err_avg = np.sum(rel_err)/n_tstep

              print("Vertical displacement on node: ", nodes[i_node])
              print("Absolute maximum error: ", abs_err_max)
              print("Absolute average error: ", abs_err_avg)
              print("Relative maximum error: ", rel_err_max)
              print("Relative average error: ", rel_err_avg)

              if abs_err_max > 0.04 or abs_err_avg > 0.012:
                     print("Absolute error limit exceeded\n")
                     if not final_compare: return False
              if rel_err_max > 1e10 or rel_err_avg > 1e10:
                     print("Relative error limit exceeded\n")
                     if not final_compare: return False  
              print("Node position within error limits\n")

       return True

# Interpolate beam spatially and temporally
def _interp_beam_t_n(beam_pos: np.ndarray, n_tstep: int, n_node: int) -> np.ndarray:
       n_tstep_old = beam_pos.shape[0]
       n_node_old = beam_pos.shape[1]

       # Interpolate spatially
       beam_pos_n = np.zeros([n_tstep_old, n_node])
       for i_ts in range(n_tstep_old):
              beam_pos_n[i_ts, :] = np.interp(np.arange(n_node), np.arange(n_node_old)/n_node_old*n_node, beam_pos[i_ts, :])

       # Interpolate temporally
       beam_pos_n_t = np.zeros([n_tstep, n_node])
       for i_n in range(n_node):
              beam_pos_n_t[:, i_n] = np.interp(np.arange(n_tstep), np.arange(n_tstep_old)/n_tstep_old*n_tstep, beam_pos_n[:, i_n])

       return beam_pos_n_t

# Interpolate velocity distribution at a point temporally
def _interp_vel_pnt_t(vel: np.ndarray, n_tstep: int): 
       n_tstep_old = vel.shape[0]
       n_pnt = vel.shape[1]

       vel_t = np.zeros([n_tstep, n_pnt, 3])

       for i_pnt in range(n_pnt):
              for i_dim in range(3):
                     vel_t[:, i_pnt, i_dim] = np.interp(np.arange(n_tstep), np.arange(n_tstep_old)/n_tstep_old*n_tstep, vel[:, i_pnt, i_dim])

       return vel_t

# Determine if the velocity at some given points has converged
# Tolerences to be set to reasonable values      
def _converge_vel_pnt(vel_prev: np.ndarray, vel_curr: np.ndarray, final_compare: bool) -> bool:
       n_tstep = np.min([vel_prev.shape[0], vel_curr.shape[0]])

       vel_prev = _interp_vel_pnt_t(vel_prev, n_tstep)
       vel_curr = _interp_vel_pnt_t(vel_curr, n_tstep)

       abs_err = np.abs(vel_prev - vel_curr)
       rel_err = np.abs(np.ones_like(vel_prev) - vel_prev/vel_curr)

       for i_pnt in range(vel_prev.shape[1]):
              print("Velocity at point: ", i_pnt)
              for i_dim in range(3):
                     print("Direction: ", ['X', 'Y', 'Z'][i_dim])

                     abs_err_max = np.max(abs_err[:, i_pnt, i_dim])
                     abs_err_avg = np.sum(abs_err[:, i_pnt, i_dim])/n_tstep

                     rel_err_max = np.max(rel_err[:, i_pnt, i_dim])
                     rel_err_avg = np.sum(rel_err[:, i_pnt, i_dim])/n_tstep
                     
                     print("Absolute maximum error: ", abs_err_max)
                     print("Absolute average error: ", abs_err_avg)
                     print("Relative maximum error: ", rel_err_max)
                     print("Relative average error: ", rel_err_avg)

                     if abs_err_max > 0.35 or abs_err_avg > 0.10:
                            print("Absolute error limit exceeded\n")
                            if not final_compare: return False
                     if rel_err_max > 1e10 or rel_err_avg > 1e10:
                            print("Relative error limit exceeded\n")
                            if not final_compare: return False  
                     print("Velocity within error limits\n")

       return True

# Extract array of vertical beam node positions over time
def _extract_beam_pos(case_data) -> np.ndarray:
        n_tstep = case_data.ts
        beam_pos = np.zeros([n_tstep, case_data.structure.timestep_info[0].pos.shape[0]])
        for i_ts in range(n_tstep):
              beam_pos[i_ts, :] = case_data.structure.timestep_info[i_ts].pos[:, 2]
        return beam_pos
