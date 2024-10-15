import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import configobj

import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils

"""
Generate a parametric swept-tip Goland wing

Input arguments:
case_name           String name for the case, vary this for each batch running case
flow                List of the solvers in order for SHARPy to run    

Outputs:
wing                Object which stores all required input parameters for SHARPy
.H5 files           Generating a wing object will automatically create the three required .H5 files

Wing discretisation:
The wing can be discretised in a variety of ways by setting the disc_mode argument
0 - Regular straight Goland wing. Any mode with a sufficiently low hinge angle will be set to this
1 - Constant chord with span reduction  
2 - Constant chord with no span reduction (correct area)
3 - Variable chord with flat tip
4 - Variable chord with pointed tip
5 - Continuous swept panel
"""


class swept_tip_goland:
    def __init__(self, case_name: str, flow: list, **kwargs):

        # Load main parameters
        self.case_name = case_name
        self.flow = flow

        # Basic IO parameters
        self.route_dir = os.path.dirname(os.path.realpath(__file__))
        self.route = self.route_dir + '/cases/'
        self.write_screen = kwargs.get('write_screen', 'on')

        # Intrinsic solution parameters
        self.itsc_aero_approx = kwargs.get('itsc_aero_approx', 'roger')
        self.itsc_dt_fact = kwargs.get('itsc_dt_fact', 1.0)
        self.itsc_dt = kwargs.get('itsc_dt', 0)
        self.itsc_nonlinear = kwargs.get('itsc_nonlinear', 1)
        self.itsc_q0treatment = kwargs.get('itsc_q0_treatment', 2)
        self.itsc_d2c_method = kwargs.get('itsc_d2c_method', 'foh')
        self.itsc_gravity_on = kwargs.get('gravity_on', False)

        self.aero_on = kwargs.get('aero_on', True)
        self.gust_on = kwargs.get('gust_on', False)
        self.single_dc_ts = kwargs.get('single_dc_ts', False)
        self.mode_sign_convention = kwargs.get('mode_sign_convention', True)
        self.tip_f = kwargs.get('tip_f', [0., 0., 0.])
        self.beam_dir = kwargs.get('beam_dir', 'y')

        # Load keyword parameters
        self.ang_h = kwargs.get('ang_h', 0.0)  # Angle of the wing kink/hinge in rad. Positive is swept back
        self.pos_frac_h = kwargs.get('pos_frac_h', 0.7)  # Fraction of the span from the root to the kink
        self.n_surf = kwargs.get('n_surf', 2)  # Number of surfaces (1 or 2)
        self.disc_mode = kwargs.get('disc_mode', 1)  # Type of discretisation (see above)
        self.M = kwargs.get('M', 16)  # Number of chordwise panels
        self.N = kwargs.get('N', 40) * self.n_surf  # Number of spanwise panels (input is per wing)
        self.u_inf = kwargs.get('u_inf', 60)  # Velocity (m/s)
        self.sym = kwargs.get('sym', True)  # Use UVLM symmetry condition (auto disabled for two wings)
        self.alpha = kwargs.get('alpha', np.deg2rad(3.0))
        self.beta = kwargs.get('beta', 0.0)
        self.yaw = kwargs.get('yaw', 0.0)
        self.roll = kwargs.get('roll', 0.0)
        self.sweep_beam = kwargs.get('sweep_beam', 0.0)
        self.sweep_panel = kwargs.get('sweep_panel', 0.0)
        self.ang_panel = kwargs.get('ang_panel', 0.0)
        self.rho = kwargs.get('rho', 1.225)
        self.c_ref = kwargs.get('c_ref', 1.8288)
        self.b_ref = kwargs.get('b_ref', 6.096)
        self.physical_time = kwargs.get('physical_time', 1.0)
        self.gravity_on = kwargs.get('gravity_on', True)
        self.sigma_1 = kwargs.get('sigma_1', 1.0)
        self.sigma_2 = kwargs.get('sigma_2', 1.0)
        self.main_ea = kwargs.get('main_ea', 0.33)
        self.main_cg = kwargs.get('main_cg', 0.5)
        self.n_lumped_mass = kwargs.get('n_lumped_mass', 1)
        self.ea = kwargs.get('ea', 1e9)
        self.ga = kwargs.get('ga', 1e9)
        self.gj = kwargs.get('gj', 0.987581e6)
        self.eiy = kwargs.get('eiy', 9.772211e6)
        self.eiz = kwargs.get('eiz', self.eiy * 100)
        self.m_unit = kwargs.get('m_unit', 35.71)
        self.j_tors = kwargs.get('j_tors', 8.64)
        self.node_bias = kwargs.get('node_bias', np.array([2, 0, 0, 0, 0, 0, 0]))
        self.airfoil_m = kwargs.get('airfoil_m', 0)
        self.airfoil_p = kwargs.get('airfoil_p', 0)
        self.gust_length = kwargs.get('gust_length', 0.0)
        self.gust_intensity = kwargs.get('gust_intensity', 0.0)
        self.gust_offset = kwargs.get('gust_offset', 0.0)
        self.n_modes_modal = kwargs.get('n_modes_modal', 20)
        self.lin_coords = kwargs.get('lin_coords', 'modes')
        self.lin_n_modes = kwargs.get('lin_n_modes', 20)
        self.lin_system = kwargs.get('lin_system', 'LinearAeroelastic')

        self.rom_method = kwargs.get('rom_method', 'Krylov')
        self.krylov_r = kwargs.get('krylov_r', 6)
        self.krylov_alg = kwargs.get('krylov_alg', 'mimo_rational_arnoldi')
        self.krylov_freq = kwargs.get('krylov_freq', 0)
        self.krylov_sing_side = kwargs.get('krylov_sing_side', 'observability')

        self.asym_v_min = kwargs.get('asym_v_min', 60)
        self.asym_v_max = kwargs.get('asym_v_max', 75)
        self.asym_v_num = kwargs.get('asym_v_num', 16)
        self.dt_factor = kwargs.get('dt_factor', 1)
        self.lin_tstep = kwargs.get('lin_tstep', 0)

        self.dt = self.c_ref / self.M / self.u_inf * self.dt_factor

        self.wake_cfl1 = kwargs.get('wake_cfl1', True)
        self.wake_dx1 = kwargs.get('wake_dx1', self.dt * self.u_inf)
        self.wake_ndx1 = kwargs.get('wake_ndx1', 12)
        self.wake_r = kwargs.get('wake_r', 1.1)
        self.wake_dxmax = kwargs.get('wake_dxmax', 30 * self.wake_dx1)
        self.wake_conv = kwargs.get('wake_conv', 2)

        if self.wake_cfl1:
            self.Mstar_fact = kwargs.get('Mstar_fact', 10)
        else:
            self.Mstar_fact = kwargs.get('Mstar_fact', 3)

        # Correct flow depending on number of wings
        # if self.n_surf == 1:     
        # if 'LinearAssembler' in self.flow:       self.flow.remove('LinearAssembler')
        # if 'AsymptoticStability' in self.flow:   self.flow.remove('AsymptoticStability')
        # if 'FrequencyResponse' in self.flow:     self.flow.remove('FrequencyResponse')

        # Calculate node and element counts
        self.n_elem_tot = self.N // 2
        self.n_elem_surf = self.n_elem_tot // self.n_surf
        self.n_node_surf = self.N // self.n_surf + 1
        self.n_node_tot = self.N + 1
        self.n_node_elem = 3

        # Calculate other case parameters required in case settings
        self.quat = algebra.euler2quat(np.array([self.roll, self.alpha, self.yaw]))

        if self.beam_dir == 'x':
            self.u_inf_direction = np.array([0., 1., 0.])
            self.panel_direction = np.array([0., 1., 0.])
        elif self.beam_dir == 'y':
            self.u_inf_direction = np.array([1., 0., 0.])
            self.panel_direction = np.array([1., 0., 0.])
        else:
            raise KeyError

        self.n_tstep = int(np.round(self.physical_time / self.dt)) if not self.single_dc_ts else 1

        # Make straight wing when below a threshold to prevent discretisation issues
        if abs(self.ang_h) < np.deg2rad(3):
            self.disc_mode = 0

        # Basic checks on input
        assert self.n_surf < 3 and self.n_surf > 0, "Must use 1 or two surfaces"
        assert self.N % 2 == 0, "UVLM spanwise panels must be even"
        assert self.n_elem_tot % self.n_surf == 0, "Cannot evenly distribute elements over surfaces"

        # Generate data required for simulation
        self._generate_beam_coords()
        self._generate_chord_main_ea()
        self._sweep_beam()
        self._generate_bcs()
        self._generate_mass_stiff()
        self._generate_connectivity_surf()
        self._generate_fem_params()
        self._generate_aero_params()
        self._mirror()
        self._generate_settings()

        self.clean_files()
        self.write_fem_file()
        self.write_aero_file()
        self.write_config_file()

    ### Calculated the curvilinear coordinate along the beam for the leading and trailing edge hinge kink
    def _le_te_h_eta(self):
        if abs(self.ang_h) <= 1e-5:
            return self.pos_frac_h

        if self.ang_h < 0:
            ang_corr = [0, self.ang_h]
        else:
            ang_corr = [self.ang_h, 0]

        pnts = []
        for i_edge in range(2):
            c_pos = [-self.main_ea * self.c_ref, (1 - self.main_ea) * self.c_ref][i_edge]
            grad = np.arctan(self.ang_h)
            x_int_corr = c_pos * (np.cos(self.ang_h) + grad * np.sin(self.ang_h))
            x_int_main = self.pos_frac_h * self.b_ref * grad
            eta_uc = (c_pos - (x_int_corr - x_int_main)) / (grad * self.b_ref)
            pnts.append(self.pos_frac_h + (eta_uc - self.pos_frac_h) * np.cos(ang_corr[i_edge]))
        return pnts

    # Calculate the curvilinear coordinate along the beam for the non-tip corner
    def _ntc_eta(self):
        if abs(self.ang_h) <= 1e-5:
            return

        if self.ang_h > 0:
            return 1 - np.arctan(self.ang_h) * ((1 - self.main_ea) * self.c_ref) / self.b_ref
        else:
            return 1 - np.arctan(self.ang_h) * (self.main_ea * self.c_ref) / self.b_ref

    ### Generate[x, y, z] coordinates with node bias
    def _generate_beam_coords(self):
        eta_lin = np.linspace(0, 1, self.n_node_surf)

        w_nd = np.zeros([self.n_node_surf])
        h_nd = np.zeros([self.n_node_surf])
        t_nd = np.zeros([self.n_node_surf])

        nd = lambda eta, y_scale, x_scale, offset: \
            y_scale * np.exp(x_scale * np.power(eta - offset, 2))

        w_nd = 1.0 + nd(eta_lin, self.node_bias[1], self.node_bias[2], 1.0)
        eta_fixed = [0.0, 1.0]

        match self.disc_mode:
            case 0:
                h_nd = nd(eta_lin, self.node_bias[3], self.node_bias[4], self.pos_frac_h)

            case 1 | 2:
                h_nd = nd(eta_lin, self.node_bias[3], self.node_bias[4], self.pos_frac_h)
                eta_fixed.append(self.pos_frac_h)

            case 3:
                h_nd = nd(eta_lin, self.node_bias[3], self.node_bias[4], self.pos_frac_h)
                eta_fixed.append(self.pos_frac_h)
                eta_fixed.extend(self._le_te_h_eta())

            case 4:
                h_nd = nd(eta_lin, self.node_bias[3], self.node_bias[4], self.pos_frac_h)
                t_nd = nd(eta_lin, self.node_bias[5], self.node_bias[6], 1)
                eta_fixed.append(self.pos_frac_h)
                eta_fixed.extend(self._le_te_h_eta())
                eta_fixed.append(self._ntc_eta())

            case 5:
                h_nd = nd(eta_lin, self.node_bias[3], self.node_bias[4], self.pos_frac_h)
                t_nd = nd(eta_lin, self.node_bias[5], self.node_bias[6], 1)
                eta_fixed.append(self.pos_frac_h)

        # Scale node spacing function to enforce predetermined nodes
        tot_nd = np.fmin(w_nd + h_nd + t_nd, self.node_bias[0])[1:]

        # Determine node numbers for predetermined nodes
        node_fix_var = []
        for i_fix in range(len(eta_fixed)):
            lin_node = int(np.round(eta_fixed[i_fix] * (self.n_node_surf - 1)))
            var_node = int(np.round(np.sum(tot_nd[:lin_node]) / np.sum(tot_nd) * (self.n_node_surf - 1)))
            if var_node in node_fix_var:
                if eta_fixed[i_fix] * (self.n_node_surf - 1) % 1.0 < 0.5:
                    var_node += 1
                else:
                    var_node -= 1
            node_fix_var.append(var_node)

        try:
            self.node_h = node_fix_var[2]
        except:
            self.node_h = None

        eta_fixed.sort()
        node_fix_var.sort()

        # Determine variable spaced node numbers for predetermined nodes
        tot_ns = np.ones_like(tot_nd) / tot_nd

        for i_sect in range(len(eta_fixed) - 1):
            curr_sum = np.sum(tot_ns[node_fix_var[i_sect]:node_fix_var[i_sect + 1]])
            des_sum = eta_fixed[i_sect + 1] - eta_fixed[i_sect]
            tot_ns[node_fix_var[i_sect]:node_fix_var[i_sect + 1]] *= des_sum / curr_sum

        # Generate node placements along beam coordinate
        self.eta = np.zeros(self.n_node_surf)
        for i_elem in range(1, self.n_node_surf):
            self.eta[i_elem] = self.eta[i_elem - 1] + tot_ns[i_elem - 1]

        # Scale to dimensional coordinates and apply hinge angle
        self.y = self.eta * self.b_ref
        self.x = np.zeros_like(self.y)
        self.z = np.zeros_like(self.y)

        if self.disc_mode in [1, 3, 4, 5]:
            self.x[self.node_h + 1:] = np.sin(self.ang_h) * (self.y[self.node_h + 1:] - self.pos_frac_h * self.b_ref)
            self.y[self.node_h + 1:] = (self.y[self.node_h + 1:] - self.pos_frac_h * self.b_ref) * np.cos(
                self.ang_h) + self.pos_frac_h * self.b_ref
        elif self.disc_mode == 2:
            self.x[self.node_h + 1:] = np.tan(self.ang_h) * (self.y[self.node_h + 1:] - self.pos_frac_h * self.b_ref)

    ### Generate chord length and elastic axis position for every beam node
    def _generate_chord_main_ea(self):

        self.panel_sweep_elem = self.sweep_panel * np.ones([self.n_elem_surf, 3])
        match self.disc_mode:
            case 0:
                self.x_le = self.x - self.c_ref * self.main_ea
                self.x_te = self.x + self.c_ref * (1 - self.main_ea)

            case 1:
                self.x_le = self._edge_1_kink(self.pos_frac_h * self.b_ref, -self.c_ref * self.main_ea)
                self.x_te = self._edge_1_kink(self.pos_frac_h * self.b_ref, self.c_ref * (1 - self.main_ea))

            case 2:
                self.x_le = self._edge_1_kink(self.pos_frac_h * self.b_ref, -self.c_ref * self.main_ea)
                self.x_te = self._edge_1_kink(self.pos_frac_h * self.b_ref, self.c_ref * (1 - self.main_ea))

            case 3:
                eta_kinks = np.array(self._le_te_h_eta())
                for i_eta in range(len(eta_kinks)):
                    if eta_kinks[i_eta] > self.pos_frac_h:
                        eta_kinks[i_eta] = self.pos_frac_h + (eta_kinks[i_eta] - self.pos_frac_h) * np.cos(self.ang_h)
                y_kinks = eta_kinks * self.b_ref

                self.x_le = self._edge_1_kink(y_kinks[0], -self.c_ref * self.main_ea)
                self.x_te = self._edge_1_kink(y_kinks[1], self.c_ref * (1 - self.main_ea))

            case 4:
                eta_kinks = np.zeros(3)
                eta_kinks[:2] = self._le_te_h_eta()
                eta_kinks[-1] = float(self._ntc_eta())
                for i_eta in range(len(eta_kinks)):
                    if eta_kinks[i_eta] > self.pos_frac_h:
                        eta_kinks[i_eta] = self.pos_frac_h + (eta_kinks[i_eta] - self.pos_frac_h) * np.cos(self.ang_h)

                y_kinks = eta_kinks * self.b_ref  #TODO: fix for swept forward wings

                if self.ang_h > 0:
                    self.x_le = self._edge_1_kink(y_kinks[0], -self.c_ref * self.main_ea)
                    self.x_te = self._edge_2_kink(y_kinks[1], y_kinks[2], self.c_ref * (1 - self.main_ea))
                else:
                    self.x_le = self._edge_2_kink(y_kinks[0], y_kinks[2], -self.c_ref * self.main_ea)
                    self.x_te = self._edge_1_kink(y_kinks[1], self.c_ref * (1 - self.main_ea))

            case 5:
                panel_sweep_nodal = np.zeros(self.n_node_surf)
                panel_sweep_nodal[:self.node_h + 1] = -(self.eta[:self.node_h + 1] / self.eta[self.node_h]) * (
                            self.ang_h / 2)
                panel_sweep_nodal[self.node_h:] = -(
                            (self.eta[self.node_h:] - self.eta[self.node_h]) / (1 - self.eta[self.node_h]) + 1) * (
                                                              self.ang_h / 2)

                relative_sweep_nodal = np.copy(panel_sweep_nodal)
                relative_sweep_nodal[self.node_h:] = -self.ang_h - relative_sweep_nodal[self.node_h:]
                self.chord_nodal = self.c_ref / np.cos(relative_sweep_nodal)

                self.chord = np.zeros([self.n_elem_surf, 3])
                self.elastic_axis = self.main_ea * np.ones([self.n_elem_surf, 3])

                for i_elem in range(self.n_elem_surf):
                    self.panel_sweep_elem[i_elem, 0] = panel_sweep_nodal[i_elem * 2]
                    self.panel_sweep_elem[i_elem, 1] = panel_sweep_nodal[i_elem * 2 + 2]
                    self.panel_sweep_elem[i_elem, 2] = panel_sweep_nodal[i_elem * 2 + 1]
                    self.chord[i_elem, 0] = self.chord_nodal[i_elem * 2]
                    self.chord[i_elem, 1] = self.chord_nodal[i_elem * 2 + 2]
                    self.chord[i_elem, 2] = self.chord_nodal[i_elem * 2 + 1]

        if self.disc_mode in [0, 1, 2, 3, 4]:
            self.chord = np.ones([self.n_elem_surf, 3])
            self.elastic_axis = np.ones([self.n_elem_surf, 3])
            for i_elem in range(self.n_elem_surf):
                for j_elem in range(3):  #TODO: fix chord and ea values to be in 0, 2, 1 order
                    self.chord[i_elem, j_elem] *= (self.x_te[2 * i_elem + j_elem] - self.x_le[2 * i_elem + j_elem])
                    self.elastic_axis[i_elem, j_elem] *= (self.x[2 * i_elem + j_elem] - \
                                                          self.x_le[2 * i_elem + j_elem]) / self.chord[i_elem, j_elem]

            self.chord_nodal = self.x_te - self.x_le
            self.elastic_axis_nodal = (self.x - self.x_le) / self.chord_nodal

    ### Sweep whole wing
    def _sweep_beam(self):
        x_new = np.cos(self.sweep_beam) * self.x + np.sin(self.sweep_beam) * self.y
        y_new = np.cos(self.sweep_beam) * self.y - np.sin(self.sweep_beam) * self.x

        self.x = x_new
        self.y = y_new

    ### Generate boundary conditions
    def _generate_bcs(self):
        self.boundary_conditions = np.zeros(self.n_node_surf, dtype=int)
        self.boundary_conditions[0] = 1
        self.boundary_conditions[-1] = -1

    ### Mirror beam nodes, chord and elastic axis values if using two surfaces
    def _mirror(self):
        if self.n_surf == 2:
            self.x = np.append(self.x, np.flip(self.x[1:]))
            self.y = np.append(self.y, -np.flip(self.y[1:]))
            self.z = np.append(self.z, np.flip(self.z[1:]))

            self.chord = np.append(self.chord, np.flip(self.chord), axis=0)
            self.elastic_axis = np.append(self.elastic_axis, np.flip(self.elastic_axis), axis=0)

            self.boundary_conditions = np.append(self.boundary_conditions, \
                                                 np.flip(self.boundary_conditions[1:]))

            self.panel_sweep_elem = np.append(self.panel_sweep_elem,
                                              -np.flip(self.panel_sweep_elem), axis=0)

    ### X position of nodes on leading or trailing edge with one kink
    def _edge_1_kink(self, y_kink, offset):
        x_root = offset * np.ones(self.n_node_surf)
        x_tip = np.tan(self.ang_h) * (self.y - y_kink) + offset

        if self.ang_h > 0:
            return np.maximum(x_root, x_tip)
        else:
            return np.minimum(x_root, x_tip)

    ### X position of nodes on leading or trailing edge with two kinks
    def _edge_2_kink(self, y_kink1, y_kink2, offset):
        x_root = offset * np.ones(self.n_node_surf)
        x_mid = np.tan(self.ang_h) * (self.y - y_kink1) + offset
        x_tip = -1 / np.tan(self.ang_h) * (self.y - y_kink2) + offset + (y_kink2 - y_kink1) * np.tan(self.ang_h)

        if self.ang_h > 0:
            return np.minimum(np.maximum(x_root, x_mid), x_tip)
        else:
            return np.maximum(np.minimum(x_root, x_mid), x_tip)

    ### Generate mass and stiffness matrices
    def _generate_mass_stiff(self):
        self.stiffness = np.zeros((1, 6, 6))
        # self.stiffness[0, :, :] = np.diag([self.sigma_2*self.ea,
        #                                     self.sigma_2*self.ga, 
        #                                     self.sigma_2*self.ga,
        #                                     self.sigma_1*self.gj, 
        #                                     self.sigma_1*self.eiy, 
        #                                     self.sigma_1*self.eiz])

        self.stiffness[0, :, :] = np.diag([self.sigma_2 * self.ea,
                                           self.sigma_2 * self.ga,
                                           self.sigma_2 * self.ga,
                                           self.sigma_1 * self.gj,
                                           self.sigma_1 * self.eiy,
                                           self.sigma_1 * self.eiz])

        pos_cg_b = np.array([0., self.c_ref * (self.main_cg - self.main_ea), 0.])
        m_chi_cg = algebra.skew(self.m_unit * pos_cg_b)
        self.mass = np.zeros((1, 6, 6))
        self.mass[0, :, :] = np.diag([self.m_unit, self.m_unit, self.m_unit,
                                      self.j_tors, 0.1 * self.j_tors, 0.9 * self.j_tors])

        # self.mass[0, :, :] = np.diag([self.m_unit, self.m_unit, self.m_unit,
        #                         self.j_tors, self.j_tors, self.j_tors])

        self.mass[0, :3, 3:] = m_chi_cg
        self.mass[0, 3:, :3] = -m_chi_cg

        self.elem_stiffness = np.zeros((self.n_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.n_elem_tot,), dtype=int)

    ### Generate connectivity and surface numbering
    def _generate_connectivity_surf(self):
        conn_loc = np.array([0, 2, 1], dtype=int)
        self.conn_surf = np.zeros([self.n_elem_surf, self.n_node_elem], dtype=int)
        self.conn_glob = np.zeros([self.n_elem_tot, self.n_node_elem], dtype=int)

        for i_elem in range(self.n_elem_surf):
            self.conn_surf[i_elem, :] = conn_loc + i_elem * (self.n_node_elem - 1)

        self.conn_glob[:self.n_elem_surf, :] = self.conn_surf
        for i_surf in range(1, self.n_surf):
            self.conn_glob[i_surf * self.n_elem_surf:(i_surf + 1) * self.n_elem_surf, :] = \
                self.conn_surf + i_surf * (self.n_node_surf - 1) + 1
            self.conn_glob[(i_surf + 1) * self.n_elem_surf - 1, 1] = 0

        self.surf_n = np.zeros(self.n_elem_tot, dtype=int)
        for i_surf in range(self.n_surf):
            self.surf_n[i_surf * self.n_elem_surf:(i_surf + 1) * self.n_elem_surf] = i_surf

    ### Generate all remaining FEM parameters
    def _generate_fem_params(self):
        if self.beam_dir == 'x':
            self.frame_of_reference_delta = np.tile([0., 1., 0.], (self.n_elem_tot, self.n_node_elem, 1))
        elif self.beam_dir == 'y':
            self.frame_of_reference_delta = np.tile([-1., 0., 0.], (self.n_elem_tot, self.n_node_elem, 1))
            # self.frame_of_reference_delta = np.tile([0., 0., -1.], (self.n_elem_tot, self.n_node_elem, 1))
        else:
            raise KeyError

        self.app_forces = np.zeros([self.n_node_tot, 6])
        self.app_forces[-1, :3] = self.tip_f

        self.lumped_mass = np.zeros(self.n_lumped_mass)
        self.lumped_mass_inertia = np.zeros([self.n_lumped_mass, 3, 3])
        self.lumped_mass_nodes = np.zeros(self.n_lumped_mass, dtype=int)
        self.lumped_mass_position = np.zeros([self.n_lumped_mass, 3])
        self.structural_twist = np.zeros([self.n_elem_tot, 3])

    ### Generate all remaining aerodynamic parameters
    def _generate_aero_params(self):
        self.airfoils_surf = []
        self.airfoils_surf.append(np.column_stack(geo_utils.generate_naca_camber(self.airfoil_m, self.airfoil_p)))
        self.airfoil_distribution = np.zeros([self.n_elem_tot, 3], dtype=int)

        self.twist = np.zeros([self.n_elem_tot, 3])
        self.surface_m = self.M * np.ones(self.n_surf, dtype=int)
        self.aero_node = np.ones(self.n_node_tot, dtype=bool)

    ### Plot shape of wing  
    def plot_wing(self):
        [_, ax] = plt.subplots(1, 1)

        # Plot chordwise vertices
        for i_node in range(len(self.y)):
            ax.plot([self.y[i_node], self.y[i_node]], \
                    [self.x[i_node] - self.chord_nodal[i_node] * self.elastic_axis_nodal[i_node], \
                     self.x[i_node] + self.chord_nodal[i_node] * (1 - self.elastic_axis_nodal[i_node])], 'k')

        # Plot spanwise vertices
        for i_x_pos in np.linspace(0, 1, self.M + 1):
            x_pos = np.zeros(len(self.y))
            for i_node in range(len(self.y)):
                x_pos[i_node] = self.x[i_node] + (i_x_pos - self.elastic_axis_nodal[i_node]) * self.chord_nodal[i_node]
            ax.plot(self.y, x_pos, 'k')

        ax.plot(self.y, self.x, 'r')
        ax.plot(self.y, self.x_le, 'b')
        ax.plot(self.y, self.x_te, 'b')
        plt.vlines(self.b_ref, -10, 10, 'k', 'dashed')
        ax.axis('equal')
        plt.xlim(-0.5, self.b_ref + 0.5)
        plt.ylim(np.min([self.x_le, self.x_te]) - 0.5, np.max([self.x_le, self.x_te]) + 0.5)
        plt.xlabel("Y (m)")
        plt.ylabel("X (m)")
        plt.show()

    ### Plot discretisation of beam
    def plot_beam_nodes(self):
        [_, ax] = plt.subplots(1, 1)
        ax.plot(self.y, self.x, '.')
        plt.vlines(self.b_ref, -10, 10, 'k', 'dashed')
        ax.axis('equal')
        plt.xlim(-0.5, self.b_ref + 0.5)
        plt.ylim(np.min([self.x_le, self.x_te]) - 0.5, np.max([self.x_le, self.x_te]) + 0.5)
        plt.xlabel("Y (m)")
        plt.ylabel("X (m)")
        plt.show()

    ### Generate FEM H5 file
    def write_fem_file(self):
        with h5.File(self.route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            if self.beam_dir == 'x':
                h5file.create_dataset('coordinates', data=np.column_stack((self.y, self.x, self.z)))
            elif self.beam_dir == 'y':
                h5file.create_dataset('coordinates', data=np.column_stack((self.x, self.y, self.z)))
            else:
                raise ValueError

            h5file.create_dataset('connectivities', data=self.conn_glob)
            h5file.create_dataset('num_node_elem', data=self.n_node_elem)
            h5file.create_dataset('num_node', data=self.n_node_tot)
            h5file.create_dataset('num_elem', data=self.n_elem_tot)
            h5file.create_dataset('stiffness_db', data=self.stiffness)
            h5file.create_dataset('elem_stiffness', data=self.elem_stiffness)
            h5file.create_dataset('mass_db', data=self.mass)
            h5file.create_dataset('elem_mass', data=self.elem_mass)
            h5file.create_dataset('frame_of_reference_delta', data=self.frame_of_reference_delta)
            h5file.create_dataset('structural_twist', data=self.structural_twist)
            h5file.create_dataset('boundary_conditions', data=self.boundary_conditions)
            h5file.create_dataset('beam_number', data=self.surf_n)
            h5file.create_dataset('app_forces', data=self.app_forces)
            h5file.create_dataset('lumped_mass', data=self.lumped_mass)
            h5file.create_dataset('lumped_mass_inertia', data=self.lumped_mass_inertia)
            h5file.create_dataset('lumped_mass_position', data=self.lumped_mass_position)
            h5file.create_dataset('lumped_mass_nodes', data=self.lumped_mass_nodes)

    ### Generate aerodynamics H5 file
    def write_aero_file(self):
        with h5.File(self.route + '/' + self.case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            for i_af in range(len(self.airfoils_surf)):
                airfoils_group.create_dataset('%d' % i_af, data=self.airfoils_surf[i_af])

            h5file.create_dataset('chord', data=self.chord).attrs['units'] = 'm'
            h5file.create_dataset('twist', data=self.twist).attrs['units'] = 'rad'
            h5file.create_dataset('airfoil_distribution', data=self.airfoil_distribution)
            h5file.create_dataset('surface_distribution', data=self.surf_n)
            h5file.create_dataset('surface_m', data=self.surface_m)
            h5file.create_dataset('m_distribution', data='uniform'.encode('ascii', 'ignore'))
            h5file.create_dataset('aero_node', data=self.aero_node)
            h5file.create_dataset('elastic_axis', data=self.elastic_axis)
            h5file.create_dataset('control_surface_type', data=[])
            h5file.create_dataset('control_surface_deflection', data=[])
            h5file.create_dataset('control_surface_chord', data=[])
            h5file.create_dataset('control_surface_hinge_coord', data=[])
            h5file.create_dataset('sweep', data=self.panel_sweep_elem)

    ### Generate settings H5 file
    def write_config_file(self):
        file_name = self.route + '/' + self.case_name + '.sharpy'
        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in self.settings.items():
            config[k] = v
        config.write()

    ### Delete any existing H5 files of the same name     
    def clean_files(self):
        try:
            os.makedirs(self.route)
        except FileExistsError:
            pass

        fem_file_name = self.route + '/' + self.case_name + '.fem.h5'
        if os.path.isfile(fem_file_name):
            os.remove(fem_file_name)
        aero_file_name = self.route + '/' + self.case_name + '.aero.h5'
        if os.path.isfile(aero_file_name):
            os.remove(aero_file_name)
        solver_file_name = self.route + '/' + self.case_name + '.sharpy'
        if os.path.isfile(solver_file_name):
            os.remove(solver_file_name)
        flightcon_file_name = self.route + '/' + self.case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)
        lininput_file_name = self.route + '/' + self.case_name + '.lininput.h5'
        if os.path.isfile(lininput_file_name):
            os.remove(lininput_file_name)
        rom_file = self.route + '/' + self.case_name + '.rom.h5'
        if os.path.isfile(rom_file):
            os.remove(rom_file)

    def _generate_settings(self):
        settings = dict()
        settings['SHARPy'] = {
            'flow': self.flow,
            'case': self.case_name,
            'route': self.route,
            'write_screen': self.write_screen,
            'write_log': 'on',
            'log_folder': self.route_dir + '/output/',
            'log_file': self.case_name + '.log',
            'route': self.route_dir + '/cases/'}

        settings['BeamLoader'] = {
            'unsteady': 'on',
            'orientation': self.quat}

        settings['AerogridLoader'] = {
            'unsteady': 'on',
            'aligned_grid': True,
            'mstar': int(self.Mstar_fact * self.M),
            'freestream_dir': self.panel_direction,
            'wake_shape_generator': 'StraightWake',
            'wake_shape_generator_input': {'u_inf': self.u_inf,
                                           'u_inf_direction': self.u_inf_direction,
                                           'dt': self.dt}}
        if not self.wake_cfl1:
            settings['AerogridLoader']['wake_shape_generator_input'].update({
                'dx1': self.wake_dx1,
                'ndx1': self.wake_ndx1,
                'r': self.wake_r,
                'dxmax': self.wake_dxmax})

        settings['StaticUvlm'] = {
            'print_info': True,
            'horseshoe': False,
            'num_cores': 8,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {'u_inf': self.u_inf,
                                     'u_inf_direction': self.u_inf_direction},
            'rho': self.rho,
            'cfl1': self.wake_cfl1}

        settings['NonLinearStatic'] = {'print_info': 'off',
                                       'max_iterations': 150,
                                       'num_load_steps': 4,
                                       'delta_curved': 1e-1,
                                       'min_delta': 1e-10,
                                       'gravity_on': self.gravity_on,
                                       'gravity': 9.81}

        settings['BeamPlot'] = {'include_rbm': 'off',
                                'include_applied_forces': 'on'}

        settings['StaticCoupled'] = {
            'print_info': 'on',
            'max_iter': 200,
            'n_load_steps': 1,
            'tolerance': 1e-15,
            'relaxation_factor': 0,
            'aero_solver': 'StaticUvlm',
            'aero_solver_settings': {
                # 'symmetry_condition': bool(self.n_surf % 2) and self.sym,
                # 'symmetry_plane': 1,
                'rho': self.rho,
                'print_info': 'off',
                'horseshoe': 'off',
                'num_cores': 4,
                'n_rollup': 0,
                'rollup_dt': self.dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'SteadyVelocityField',
                'velocity_field_input': {
                    'u_inf': self.u_inf,
                    'u_inf_direction': self.u_inf_direction}},
            'structural_solver': 'NonLinearStatic',
            'structural_solver_settings': {'print_info': 'off',
                                           'max_iterations': 150,
                                           'num_load_steps': 4,
                                           'delta_curved': 1e-1,
                                           'min_delta': 1e-10,
                                           'gravity_on': self.gravity_on,
                                           'gravity': 9.81}}

        settings['DynamicCoupled'] = {'structural_solver': 'NonLinearDynamicPrescribedStep',
                                      'structural_solver_settings': {'print_info': 'off',
                                                                     'max_iterations': 950,
                                                                     'delta_curved': 1e-1,
                                                                     'newmark_damp': 5e-3,
                                                                     'min_delta': 1e-5,
                                                                     'abs_threshold': 1e-13,
                                                                     'gravity_on': self.gravity_on,
                                                                     'gravity': 9.81,
                                                                     'num_steps': self.n_tstep,
                                                                     'dt': self.dt,
                                                                     },
                                      # 'aero_solver': 'NoAero',
                                      # 'aero_solver_settings': {},
                                      'aero_solver': 'StepUvlm' if self.aero_on else 'NoAero',
                                      'aero_solver_settings': {'print_info': 'off',
                                                               'num_cores': 8,
                                                               'rho': self.rho,
                                                               # 'symmetry_condition': bool(self.n_surf % 2) and self.sym,
                                                               # 'symmetry_plane': 1,
                                                               'convection_scheme': self.wake_conv,
                                                               'gamma_dot_filtering': 6,
                                                               'n_time_steps': self.n_tstep,
                                                               'dt': self.dt,
                                                               'cfl1': self.wake_cfl1,
                                                               'velocity_field_generator': 'GustVelocityField',
                                                               'velocity_field_input': {'u_inf': self.u_inf,
                                                                                        'u_inf_direction': self.u_inf_direction,
                                                                                        'gust_shape': '1-cos',
                                                                                        'gust_parameters': {
                                                                                            'gust_length': self.gust_length,
                                                                                            'gust_intensity': self.gust_intensity * self.u_inf},
                                                                                        'offset': self.gust_offset,
                                                                                        'relative_motion': 'on'}} if self.aero_on else {},
                                      # 'velocity_field_generator': 'SteadyVelocityField',
                                      # 'velocity_field_input': {'u_inf': self.u_inf,
                                      #     'u_inf_direction': self.u_inf_direction}},
                                      'fsi_substeps': 200,
                                      'minimum_steps': 1,
                                      'relaxation_steps': 150,
                                      'final_relaxation_factor': 0.5,
                                      'n_time_steps': self.n_tstep,
                                      'dt': self.dt,
                                      'include_unsteady_force_contribution': 'on',
                                      'postprocessors': ['BeamLoads', 'BeamPlot', 'AerogridPlot'],
                                      'postprocessors_settings': {'BeamLoads': {'csv_output': 'off'},
                                                                  'BeamPlot': {'include_rbm': 'on',
                                                                               'include_applied_forces': 'on'},
                                                                  'AerogridPlot': {
                                                                      'include_rbm': 'on',
                                                                      'include_applied_forces': 'on',
                                                                      'minus_m_star': 0},
                                                                  }}

        settings['Modal'] = {'NumLambda': self.n_modes_modal,
                             'rigid_body_modes': 'off',
                             'print_matrices': 'on',
                             'rigid_modes_cg': 'off',
                             'continuous_eigenvalues': 'off',
                             'dt': self.dt,
                             'plot_eigenvalues': False,
                             'max_rotation_deg': 15.,
                             'max_displacement': 0.15,
                             # 'max_rotation_deg': 0.0,
                             # 'max_displacement': 0.0,
                             'write_modes_vtk': True,
                             'use_undamped_modes': True,
                             'mode_sign_convention': self.mode_sign_convention}

        if self.lin_system == 'LinearAeroelastic':
            settings['LinearAssembler'] = {
                'linear_system': 'LinearAeroelastic',
                'linearisation_tstep': self.lin_tstep,
                'modal_tstep': 0,
                'inout_coordinates': self.lin_coords,  # 'nodes', 'modes'
                'linear_system_settings': {
                    'beam_settings': {'modal_projection': 'on',
                                      'inout_coords': 'modes',
                                      'discrete_time': 'on',
                                      'newmark_damp': 0.5e-4,
                                      'discr_method': 'newmark',
                                      'dt': self.dt,
                                      'proj_modes': 'undamped',
                                      'use_euler': 'off',
                                      'num_modes': self.lin_n_modes,
                                      'print_info': 'on',
                                      'gravity': 'on',
                                      'remove_sym_modes': 'off',
                                      'remove_dofs': []},
                    'aero_settings': {'dt': self.dt,
                                      'density': self.rho,
                                      #   'gust_assembler': 'LeadingEdge',
                                      # use scalingdict to put in reduced time s, not for t
                                      # 'ScalingDict': {'length': 0.5 * self.c_ref,
                                      #                 'speed': self.u_inf,
                                      #                 'density': self.rho},
                                      'integr_order': 2,
                                      'density': self.rho,
                                      'remove_predictor': 'on',
                                      'use_sparse': 'on',
                                      # 'remove_inputs': ['u_gust'],
                                      }
                }}

            match self.rom_method:
                case 'Krylov':
                    settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method'] = [
                        self.rom_method]
                    settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method_settings'] = \
                        {'Krylov': {'r': self.krylov_r,
                                    'algorithm': self.krylov_alg,
                                    'frequency': self.krylov_freq,
                                    'single_side': self.krylov_sing_side}}
                case 'Balanced':
                    settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method'] = [
                        self.rom_method]
                    settings['LinearAssembler']['linear_system_settings']['aero_settings']['rom_method_settings'] = \
                        {'Balanced': {'algorithm': 'FrequencyLimited',
                                      'algorithm_settings': {'frequency': 1.5}}}

        elif self.lin_system == 'LinearBeam':
            settings['LinearAssembler'] = {'linear_system': 'LinearBeam',
                                           'linearisation_tstep': self.lin_tstep,
                                           'modal_tstep': 0,
                                           'inout_coordinates': self.lin_coords,  # 'nodes', 'modes'
                                           'linear_system_settings': {'modal_projection': 'on',
                                                                      'inout_coords': 'modes',
                                                                      'discrete_time': 'on',
                                                                      'newmark_damp': 0.5e-4,
                                                                      'discr_method': 'newmark',
                                                                      'dt': self.dt,
                                                                      'proj_modes': 'undamped',
                                                                      'use_euler': 'off',
                                                                      'num_modes': self.lin_n_modes,
                                                                      'print_info': 'on',
                                                                      'gravity': 'on',
                                                                      'remove_sym_modes': 'off',
                                                                      'remove_dofs': []}}

        settings['AsymptoticStability'] = {'print_info': True,
                                           'velocity_analysis': [self.asym_v_min, self.asym_v_max, self.asym_v_num],
                                           'modes_to_plot': []}

        settings['LinearRFA'] = {'p_min': 1,
                                 'p_max': 6.0,
                                 'p_num': 1,
                                 'p_spacing': 'geometric',
                                 'p_pow_min': 1.0,
                                 'p_pow_max': 1.5,
                                 'num_poles': 8,
                                 'grad_opt': False,
                                 'plot_rfa': False,
                                 'num_q_plot': 2,
                                 'plot_type': 'bode',
                                 'k_min': 1e-4,
                                 'k_max': 1.50,
                                 'k_num': 100,
                                 'err_type': 'norm_norm',
                                 'fit_u_gust': True,
                                 'minimum_promote': 1e-1,
                                 'force_negative': True,
                                 'k_inf': [],
                                 'pole_duplicate': False,
                                 'pole_duplicate_offset': -0.8,
                                 'k_extended_min': 1e-4,
                                 'k_extended_max': 1e9,
                                 'k_extended_num': 800,
                                 'ScalingDict': {'length': 0.5 * self.c_ref,
                                                 'speed': self.u_inf,
                                                 'density': self.rho}
                                 }

        settings['Intrinsic'] = {'num_modes': self.lin_n_modes,
                                 'orientation': self.quat,
                                 'd2c_method': self.itsc_d2c_method,
                                 'aero_approx': self.itsc_aero_approx,
                                 't1': self.physical_time,
                                 'dt': self.dt * self.itsc_dt_fact,
                                 'solution': 'dynamic',
                                 'c_ref': self.c_ref,
                                 'rho': self.rho,
                                 'u_inf': self.u_inf,
                                 'gravity_on': self.itsc_gravity_on,
                                 'aero_on': self.aero_on,
                                 'nonlinear_structure': self.itsc_nonlinear,
                                 'q0_treatment': self.itsc_q0treatment,
                                 'gust_on': self.gust_on,
                                 'gust_intensity': self.gust_intensity * self.u_inf,
                                 'gust_offset': self.gust_offset,
                                 'gust_length': self.gust_length,
                                 'gust_num_x': 30,
                                 'tip_force': self.tip_f}

        settings['IntrinsicPlot'] = {'stride': int(1. / self.itsc_dt_fact)}

        self.settings = settings
