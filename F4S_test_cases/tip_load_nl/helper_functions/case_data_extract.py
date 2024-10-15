import numpy as np
import os
import fnmatch
import pandas


def case_data_extract(wing, case_data):
    dict_out = {'ang_h': wing.ang_h,
                'pos_frac_h': wing.pos_frac_h,
                'dt': wing.dt,
                'u_inf': wing.u_inf,
                'M': wing.M,
                'N': wing.N,
                'c_ref': wing.c_ref,
                'b_ref': wing.b_ref,
                'gust_intensity': wing.gust_intensity,
                'gust_length': wing.gust_length,
                'gust_offset': wing.gust_offset,
                'disc_mode': wing.disc_mode,
                'x': wing.x,
                'y': wing.y,
                'z': wing.z,
                'n_surf': wing.n_surf,
                'n_elem_tot': wing.n_elem_tot,
                'n_elem_surf': wing.n_elem_surf,
                'n_node_surf': wing.n_node_surf,
                'n_node_tot': wing.n_node_tot,
                'n_node_elem': wing.n_node_elem,
                'case_name': wing.case_name,
                'physical_time': wing.physical_time,
                'n_modes': wing.lin_n_modes,
                }

    if wing.node_h is None:
        dict_out.update({'node_h': -1})
    else:
        dict_out.update({'node_h': wing.node_h})

    n_tstep = len(case_data.structure.timestep_info)

    beam_pos = np.zeros([n_tstep, wing.n_node_tot, 3])
    beam_pos_dot = np.zeros_like(beam_pos)
    q = np.zeros([n_tstep, case_data.structure.timestep_info[0].q.shape[0]])

    for i_ts in range(n_tstep):
        beam_pos[i_ts, :, :] = case_data.structure.timestep_info[i_ts].pos
        beam_pos_dot[i_ts, :, :] = case_data.structure.timestep_info[i_ts].pos_dot
        q[i_ts, :] = case_data.structure.timestep_info[i_ts].q

    beam_pos_init = case_data.structure.ini_info.pos
    psi_init = case_data.structure.ini_info.psi

    dict_out.update({'beam_pos': beam_pos, 'beam_pos_dot': beam_pos_dot, 'beam_pos_init': beam_pos_init, \
                     'psi_init': psi_init, 'n_tstep': n_tstep, 'q': q})

    zeta = np.zeros([n_tstep] + list(case_data.aero.timestep_info[0].zeta[0].shape))
    gamma = np.zeros([n_tstep] + list(case_data.aero.timestep_info[0].gamma[0].shape))
    zeta_star = np.zeros([n_tstep] + list(case_data.aero.timestep_info[0].zeta_star[0].shape))
    gamma_star = np.zeros([n_tstep] + list(case_data.aero.timestep_info[0].gamma_star[0].shape))
    psi = np.zeros([n_tstep] + list(case_data.structure.timestep_info[0].psi.shape))

    for i_t in range(n_tstep):
        zeta[i_t, :, :] = case_data.aero.timestep_info[i_t].zeta[0]
        gamma[i_t, :, :] = case_data.aero.timestep_info[i_t].gamma[0]
        zeta_star[i_t, :, :] = case_data.aero.timestep_info[i_t].zeta_star[0]
        gamma_star[i_t, :, :] = case_data.aero.timestep_info[i_t].gamma_star[0]
        psi[i_t, :, :, :] = case_data.structure.timestep_info[i_t].psi

    dict_out.update({'zeta': zeta, 'gamma': gamma, 'zeta_star': zeta_star, 'gamma_star': gamma_star, 'psi': psi})

    if 'Modal' in wing.flow:
        mode_freqs = case_data.structure.timestep_info[0].modal['freq_natural']
        eigenvalues = case_data.structure.timestep_info[0].modal['eigenvalues']
        eigenvectors = case_data.structure.timestep_info[0].modal['eigenvectors']
        M = case_data.structure.timestep_info[0].modal['M']
        K = case_data.structure.timestep_info[0].modal['K']

        modal_dict = {'mode_freqs': mode_freqs, 'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'M': M,
                      'K': K}
        dict_out.update({"Modal": modal_dict})

    # if 'AeroForcesCalculator' in wing.flow:
    #     force_data = pandas.read_csv('./output/%s/forces/forces_aeroforces.txt' % wing.case_name, delimiter=', ', index_col=False).to_dict()
    #     moment_data = pandas.read_csv('./output/%s/forces/moments_aeroforces.txt' % wing.case_name, delimiter=', ', index_col=False).to_dict()

    #     forces_a_s = np.zeros([n_tstep, 3])
    #     forces_g_s = np.zeros([n_tstep, 3])
    #     moments_a_s = np.zeros([n_tstep, 3])
    #     moments_g_s = np.zeros([n_tstep, 3])
    #     forces_a_u = np.zeros([n_tstep, 2])
    #     forces_g_u = np.zeros([n_tstep, 3])
    #     moments_a_u = np.zeros([n_tstep, 3])
    #     moments_g_u = np.zeros([n_tstep, 3])

    #     for i_ts in range(n_tstep):
    #         forces_a_s[i_ts, :] = [force_data['fx_steady_a'][i_ts], force_data['fy_steady_a'][i_ts], force_data['fz_steady_a'][i_ts]]
    #         forces_g_s[i_ts, :] = [force_data['fx_steady_G'][i_ts], force_data['fy_steady_G'][i_ts], force_data['fz_steady_G'][i_ts]]
    #         moments_a_s[i_ts, :] = [moment_data['mx_steady_a'][i_ts], moment_data['my_steady_a'][i_ts], moment_data['mz_steady_a'][i_ts]]
    #         moments_g_s[i_ts, :] = [moment_data['mx_steady_G'][i_ts], moment_data['my_steady_G'][i_ts], moment_data['mz_steady_G'][i_ts]]
    #         forces_a_u[i_ts, :] = [force_data['fx_unsteady_a'][i_ts], force_data['fy_unsteady_a'][i_ts]]
    #         forces_g_u[i_ts, :] = [force_data['fx_unsteady_G'][i_ts], force_data['fy_unsteady_G'][i_ts], force_data['fz_unsteady_G'][i_ts]]
    #         moments_a_u[i_ts, :] = [moment_data['mx_unsteady_a'][i_ts], moment_data['my_unsteady_a'][i_ts], moment_data['mz_unsteady_a'][i_ts]]
    #         moments_g_u[i_ts, :] = [moment_data['mx_unsteady_G'][i_ts], moment_data['my_unsteady_G'][i_ts], moment_data['mz_unsteady_G'][i_ts]]

    #     force_dict = {"moments_a_s": moments_a_s, 
    #                         "moments_g_s": moments_g_s, 
    #                         "forces_a_s": forces_a_s, 
    #                         "forces_g_s": forces_g_s,
    #                         "moments_a_u": moments_a_u, 
    #                         "moments_g_u": moments_g_u, 
    #                         "forces_a_u": forces_a_u, 
    #                         "forces_g_u": forces_g_u}
    #     dict_out.update({'AeroForcesCalculator': force_dict})

    if 'LinearAssembler' in wing.flow:
        in_head = []
        in_head_bounds = []
        for i_head in case_data.linear.ss.input_variables.vector_variables:
            in_head.append(i_head.name)
            in_head_bounds.append([i_head.first_position, i_head.end_position])

        state_head = []
        state_head_bounds = []
        for i_head in case_data.linear.ss.state_variables.vector_variables:
            state_head.append(i_head.name)
            state_head_bounds.append([i_head.first_position, i_head.end_position])

        out_head = []
        out_head_bounds = []
        for i_head in case_data.linear.ss.output_variables.vector_variables:
            out_head.append(i_head.name)
            out_head_bounds.append([i_head.first_position, i_head.end_position])

        A = case_data.linear.ss.A
        B = case_data.linear.ss.B
        C = case_data.linear.ss.C
        D = case_data.linear.ss.D
        dt_ss = case_data.linear.ss.dt

        linear_dict = {"in_head": in_head,
                       "in_head_bounds": in_head_bounds,
                       "out_head": out_head,
                       "out_head_bounds": out_head_bounds,
                       "state_head": state_head,
                       "state_head_bounds": state_head_bounds,
                       "A": A,
                       "B": B,
                       "C": C,
                       "D": D,
                       "dt_ss": dt_ss}
        dict_out.update({"Linear": linear_dict})

    # if 'AsymptoticStability' in wing.flow:
    #     for file_name in os.listdir('./output/%s/stability/' % wing.case_name):
    #         if(fnmatch.fnmatch(file_name, '*.dat')):
    #             velocity_analysis = np.loadtxt('./output/%s/stability/%s' % (wing.case_name, file_name))

    #     u_inf_out = velocity_analysis[:, 0]
    #     eigs_r = velocity_analysis[:, 1]
    #     eigs_i = velocity_analysis[:, 2]

    #     v_div = 0
    #     v_flutter = 0

    #     im_div_t = 1e-5
    #     num_modes_stab = int(len(u_inf_out)/wing.asym_v_num)

    #     for i_vel in range(wing.asym_v_num):
    #         for i_mode in range(num_modes_stab):
    #             if eigs_r[i_vel*num_modes_stab + i_mode] >= 0 and abs(eigs_i[i_vel*num_modes_stab + i_mode]) <= im_div_t:
    #                 v_div = u_inf_out[i_vel*num_modes_stab + i_mode]
    #                 break

    #     for i_vel in range(wing.asym_v_num):                                                          
    #         for i_mode in range(num_modes_stab):
    #             if eigs_r[i_vel*num_modes_stab + i_mode] >= 0 and abs(eigs_i[i_vel*num_modes_stab + i_mode]) >= im_div_t:
    #                 v_flutter = u_inf_out[i_vel*num_modes_stab + i_mode]
    #                 break

    #     n_unst_modes = np.ones(wing.asym_v_num) 
    #     for i_vel in range(wing.asym_v_num):
    #         for i_mode in range(num_modes_stab):
    #             if eigs_r[i_vel*num_modes_stab + i_mode] >= 0:
    #                 n_unst_modes[i_vel] += 1

    #     stab_dict = {
    #         'v_div': v_div,
    #         'v_flutter': v_flutter,
    #         'n_unst_modes': n_unst_modes
    #     }
    #     dict_out.update({'AsymptoticStability': stab_dict})

    if 'Intrinsic' in wing.flow:
        intrinsic_dict = {'q': np.array(case_data.intrinsic.q),
                          'ra': np.array(case_data.intrinsic.ra),
                          't': np.array(case_data.intrinsic.t),
                          'gamma1': np.array(case_data.intrinsic.gamma1),
                          'gamma2': np.array(case_data.intrinsic.gamma2),
                          'phi1': np.array(case_data.intrinsic.modes.phi1),
                          'psi1': np.array(case_data.intrinsic.modes.psi1),
                          'phi2': np.array(case_data.intrinsic.modes.phi2),
                          'C06ab': np.array(case_data.intrinsic.modes.C06ab),
                          'C0ab': np.array(case_data.intrinsic.modes.C0ab),
                          'phi1l': np.array(case_data.intrinsic.modes.phi1l),
                          'phi1ml': np.array(case_data.intrinsic.modes.phi1ml),
                          'phi2l': np.array(case_data.intrinsic.modes.phi2l),
                          'psi1l': np.array(case_data.intrinsic.modes.psi1l),
                          'psi2l': np.array(case_data.intrinsic.modes.psi2l)
                          }
        try:
            intrinsic_dict.update({
                'eta_a_jig': case_data.intrinsic.eta_a_jig,
                'f_jig': case_data.intrinsic.f_jig})
        except AttributeError:
            pass

        dict_out.update({'Intrinsic': intrinsic_dict})

    return dict_out
