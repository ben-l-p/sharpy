# Python package imports
import jax
import numpy as np
import jax.numpy as jnp
from scipy.linalg import block_diag
import pyyeti
from typing import Optional, Sequence
import warnings

# General SHARPy imports
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout
import sharpy.presharpy.presharpy
import sharpy.utils.algebra as algebra

# FEM4INAS
from fem4inas import fem4inas_main
from fem4inas.preprocessor.configuration import Config
from fem4inas.preprocessor.inputs import Inputs


@solver
class IntrinsicFlutterSolver(BaseSolver):
    """
    Solver which calls FENIAX, which computes a time-domain solution comprised of a
    non-linear structural model coupled with linearised aerodynamics. These can be included
    as either a Roger's aerodynamic approximation in the frequency domain, obtained through the
    'linearrfa' solver, or by integrating state space aerodynamics.

    The 'Modal', 'StaticUvlm' and 'LinearAssembler' solver must be run prior, as well as 'LinearRFA'
    if using Roger's aero. The 'StaticUVLM' solver is here used to get a developed flow in the UVLM
    for linearising around, whilst not allowing any structural deflections. This is due to the
    intrinsic formulation requiring an undeformed reference state.

    flow =  ['BeamLoader',
            'AerogridLoader',
            'Modal',
            'StaticUvlm',
            'LinearAssembler',
            'Intrinsic']

    """

    # Settings used to generate inputs for FEM4INAS
    solver_id = 'IntrinsicFlutter'
    solver_classification = 'Coupled'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write status to screen'

    settings_types['num_modes'] = 'int'
    settings_default['num_modes'] = None
    settings_description['num_modes'] = 'Number of modes to retain'

    settings_types['orientation'] = 'list(float)'
    settings_default['orientation'] = [1., 0., 0., 0.]
    settings_description['orientation'] = 'Quaternion used to describe rotation from inertial to body frame'

    settings_types['aero_approx'] = 'str'
    settings_default['aero_approx'] = 'statespace'
    settings_description['aero_approx'] = 'Aerodynamic model to use.'
    settings_options['aero_approx'] = ['roger', 'statespace', 'none']

    settings_types['delta_curved'] = 'float'
    settings_default['delta_curved'] = 1e-2
    settings_description['delta_curved'] = 'Threshold for linear rotations'

    settings_types['use_custom_timestep'] = 'int'
    settings_default['use_custom_timestep'] = 0
    settings_description['use_custom_timestep'] = 'Time step of structure for calculating modes'

    settings_types['component_names'] = 'list(str)'
    settings_default['component_names'] = []
    settings_description['component_names'] = ('Name components of the structure. '
                                               'Will use lettering [A, B, ...] by default')

    settings_types['d2c_method'] = 'str'
    settings_default['d2c_method'] = 'tustin'
    settings_description['d2c_method'] = 'Method for converting state space from discrete to continuous time'
    settings_options['d2c_method'] = ['zoh', 'zoha', 'foh', 'tustin']

    settings_types['engine'] = 'str'
    settings_default['engine'] = 'intrinsicmodal'
    settings_description['engine'] = 'Engine to be used in FEM4INAS'

    settings_types['driver'] = 'str'
    settings_default['driver'] = 'intrinsic'
    settings_description['driver'] = 'Driver to be used in FEM4INAS'

    settings_types['sim_type'] = 'str'
    settings_default['sim_type'] = 'single'
    settings_description['sim_type'] = 'Simulation type to be used in FEM4INAS'

    settings_types['solution'] = 'str'
    settings_default['solution'] = 'dynamic'
    settings_description['solution'] = 'Solution type to be used in FEM4INAS'

    settings_types['solver_library'] = 'str'
    settings_default['solver_library'] = 'diffrax'
    settings_description['solver_library'] = 'Solver library to be used in FEM4INAS'

    settings_types['solver_function'] = 'str'
    settings_default['solver_function'] = 'ode'
    settings_description['solver_function'] = 'Solver function to be used in FEM4INAS'

    settings_types['solver_name'] = 'str'
    settings_default['solver_name'] = 'Dopri5'
    settings_description['solver_name'] = 'Solver name to be used in FEM4INAS'

    settings_types['rtol'] = 'float'
    settings_default['rtol'] = 1e-6
    settings_description['rtol'] = 'Solver relative tolerance to be used in FEM4INAS'

    settings_types['atol'] = 'float'
    settings_default['atol'] = 1e-6
    settings_description['atol'] = 'Solver absolute tolerance to be used in FEM4INAS'

    settings_types['max_steps'] = 'int'
    settings_default['max_steps'] = 50
    settings_description['max_steps'] = 'Maximum number of steps to be used in FEM4INAS'

    settings_types['norm'] = 'str'
    settings_default['norm'] = 'linalg_norm'
    settings_description['norm'] = 'Norm function to be used in FEM4INAS'

    settings_types['kappa'] = 'float'
    settings_default['kappa'] = 1e-2
    settings_description['kappa'] = 'Kappa value to be used in FEM4INAS'

    settings_types['rho'] = 'float'
    settings_default['rho'] = None
    settings_description['rho'] = 'Freestream density (kg/m^3)'

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = None
    settings_description['u_inf'] = 'Freestream velocity (m/s)'

    settings_types['c_ref'] = 'float'
    settings_default['c_ref'] = None
    settings_description['c_ref'] = 'Reference chord (m)'

    settings_types['gravity_on'] = 'bool'
    settings_default['gravity_on'] = True
    settings_description['gravity_on'] = 'Enable gravity'

    settings_types['nonlinear_structure'] = 'int'
    settings_default['nonlinear_structure'] = 1
    settings_description['nonlinear_structure'] = 'Include nonlinear structural couplings (gamma terms)'

    settings_types['q0_treatment'] = 'int'
    settings_default['q0_treatment'] = 2
    settings_description['q0_treatment'] = 'Method for obtaining q0'

    settings_types['max_iter'] = 'int'
    settings_default['max_iter'] = 100
    settings_description['max_iter'] = 'Maximum number of iterations for finding static solution'

    settings_types['remove_symmetric'] = 'bool'
    settings_default['remove_symmetric'] = True
    settings_description['remove_symmetric'] = 'Remove symmetric modes'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.input = None
        self.evecs_ae = None
        self.evals_ae = None
        self.sys_mat = None
        self.data = Optional[sharpy.presharpy.presharpy.PreSharpy]
        self.settings: Optional[dict] = None
        self.m_global: Optional[np.ndarray] = None
        self.k_global: Optional[np.ndarray] = None
        self.tn: Optional[int] = None
        self.aero_model: Optional[str] = None
        self.x: Optional[np.ndarray] = None
        self.conn: Optional[np.ndarray] = None
        self.beam_number: Optional[np.ndarray] = None
        self.num_modes: Optional[int] = None
        self.num_nodes: Optional[int] = None
        self.node_numbers: Optional[list[int]] = None
        self.component_names: Optional[list[str]] = None
        self.node_names: Optional[list[str]] = None

        self.ss_c_time = None
        self.ss_A = None
        self.ss_B0 = None
        self.ss_B1 = None
        self.ss_C = None
        self.ss_D0 = None
        self.ss_D1 = None
        self.num_lags = None
        self.fm_jig_nodal = None
        self.fm_jig_modal = None
        self.states_keep = None
        self.outputs_keep = None

        self.gamma1: Optional[np.ndarray] = None
        self.gamma2: Optional[np.ndarray] = None
        self.phi1: Optional[np.ndarray] = None
        self.q0_bar: Optional[np.ndarray] = None
        self.q2_bar: Optional[np.ndarray] = None
        self.lambda_bar: Optional[np.ndarray] = None

    def initialise(self, data: sharpy.presharpy.presharpy.PreSharpy, custom_settings=None, restart=False):
        # Load solver settings
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)

    class IntrinsicObj:
        def __init__(self, sol, m: np.ndarray, k: np.ndarray, orientation: Sequence):
            self.M = m
            self.K = k
            self.Cab = np.array(sol.dynamicsystem_s1.Cab)
            self.X1 = np.array(sol.dynamicsystem_s1.X1)
            self.X2 = np.array(sol.dynamicsystem_s1.X2)
            self.X3 = np.array(sol.dynamicsystem_s1.X3)
            self.q = np.array(sol.dynamicsystem_s1.q)
            self.r_g = np.swapaxes(np.array(sol.dynamicsystem_s1.ra), 1, 2)
            self.r_a = np.squeeze(algebra.quat2rotation(orientation).T @ np.expand_dims(self.r_g, -1))
            self.t = np.array(sol.dynamicsystem_s1.t)

            try:
                self.eta_a_jig = np.array(
                    sol.modalaerostatespace_s1.eta_a_jig)  # TODO: make this work for other aero models
                self.f_jig = np.array(sol.modalaerostatespace_s1.f_jig)
            except AttributeError:
                pass

            self.gamma1 = np.array(sol.couplings.gamma1)
            self.gamma2 = np.array(sol.couplings.gamma2)

            self.phi1 = np.array(sol.modes.phi1)
            self.psi1 = np.array(sol.modes.psi1l)
            self.phi2 = np.array(sol.modes.phi2)
            self.psi2 = np.array(sol.modes.psi2l)
            self.omega = np.array(sol.modes.omega)

            self.modes = sol.modes

            self.flutter = dict()

    def run(self, **kwargs) -> sharpy.presharpy.presharpy.PreSharpy:
        # Create all case inputs
        self.get_grid()
        self.transform_struct()
        self.evals_struct, self.evecs_struct = self.calculate_eigs()

        # Add aero attributes from input aero model
        self.aero_model: str = self.settings['aero_approx']

        if self.aero_model == 'statespace':
            self.statespace_structure()
        else:
            raise KeyError(f"Aero model {self.aero_model} not recognised")

        # Generate case object
        self.input = self.generate_settings_file()
        config = Config(self.input)

        # Run case
        cout.cout_wrap("\nRunning Intrinsic Solver", 0)
        sol = fem4inas_main.main(input_obj=config)
        cout.cout_wrap("Intrinsic Solution Complete", 0)

        intrinsic_out = self.IntrinsicObj(sol, self.m_global, self.k_global, self.settings['orientation'])
        self.data.intrinsic = intrinsic_out

        self.phi1 = np.array(self.data.intrinsic.phi1)
        self.gamma1 = np.array(self.data.intrinsic.gamma1)
        self.gamma2 = np.array(self.data.intrinsic.gamma2)
        self.omega = np.array(self.data.intrinsic.omega)

        # if self.settings['remove_symmetric']:
        #     new_phi1 = np.zeros((self.num_modes // 2, *self.phi1.shape[1:]))
        #     for i_mode in range(self.num_modes // 2):
        #         new_phi1[i_mode, ...] = self.phi1[2 * i_mode, ...]
        #         new_phi1[i_mode, ...] += self.phi1[2 * i_mode + 1, ...]
        #     self.phi1 = new_phi1
        #     self.omega = self.omega[::2]
        #     self.gamma1 = self.gamma1[::2, ::2, ::2]
        #     self.gamma2 = self.gamma2[::2, ::2, ::2]

        self.jig_loads()

        self.data.intrinsic.flutter['r_bar'] = self.find_static_sol()
        sys_mat = self.assemble_sys()

        self.data.intrinsic.flutter['q2_bar'] = self.q2_bar
        self.data.intrinsic.flutter['q0_bar'] = self.q0_bar
        self.data.intrinsic.flutter['lambda_bar'] = self.lambda_bar
        self.data.intrinsic.flutter['sys_mat'] = sys_mat
        self.data.intrinsic.flutter['evecs_ae'] = self.evecs_ae
        self.data.intrinsic.flutter['evals_ae'] = self.evals_ae
        self.data.intrinsic.flutter['evecs_struct'] = self.evecs_struct

        return self.data

    # Returns the grid of the system
    def get_grid(self) -> None:
        # Create component names
        n_beams = self.data.structure.beam_number.max() + 1
        if self.settings['component_names']:
            assert n_beams == len(
                self.settings['component_names']), "Number of component names does not match number of components"
            self.component_names = self.settings['component_names']
        else:
            self.component_names = [chr(65 + i) for i in range(n_beams)]

        # orientation is given as quat_GA
        rot_orient = algebra.quat2rotation(self.settings['orientation'])
        # Create grid
        self.x = np.squeeze(rot_orient @ np.expand_dims(
            self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos, -1))
        self.conn = self.data.structure.connectivities  # SHARPy format
        self.beam_number = self.data.structure.beam_number  # [num_elem]

        self.num_modes = self.settings['num_modes']
        self.node_numbers = self.data.structure.global_nodes_num - 1  # take away one to make 0 the reference

        self.node_names = [self.component_names[0]] + [self.component_names[i] for i in self.beam_number for _ in
                                                       (0, 1)]

        pass

    def transform_struct(self) -> None:
        # transform mass and stiffness to global frame
        m_modal = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['M']
        k_modal = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['K']

        num_nodes = m_modal.shape[0] // 6

        psi0 = np.vstack((self.data.structure.timestep_info[self.settings['use_custom_timestep']].psi[0, 0, :],
                          self.data.structure.timestep_info[self.settings['use_custom_timestep']].psi[:, [2, 1], :]
                          .reshape(-1, 3)))

        tan_psi0 = np.apply_along_axis(algebra.crv2tan, axis=1, arr=psi0)

        rot_orient = algebra.quat2rotation(self.settings['orientation']).T

        tfrm = np.linalg.inv(block_diag(
            *[np.block([[rot_orient, np.zeros((3, 3))], [np.zeros((3, 3)), rot_orient @ tan_psi0[i + 1, ...]]])
              for i in range(num_nodes)]))

        i_tfrm = tfrm.T

        self.m_global = tfrm @ m_modal @ i_tfrm
        self.k_global = tfrm @ k_modal @ i_tfrm

    def generate_settings_file(self) -> Inputs:
        """
        Generate case file which is passed to FENIAX
        """

        inp = Inputs()

        # General settings
        inp.systems.sett.s1.t1 = 0.0
        inp.systems.sett.s1.tn = 2
        inp.systems.sett.s1.aero.rho_inf = self.settings['rho']
        inp.systems.sett.s1.aero.u_inf = self.settings['u_inf']
        inp.systems.sett.s1.aero.c_ref = self.settings['c_ref']
        inp.systems.sett.s1.xloads.modalaero_forces = False
        inp.systems.sett.s1.xloads.gravity_forces = False
        inp.systems.sett.s1.nonlinear = self.settings['nonlinear_structure']
        inp.systems.sett.s1.q0treatment = self.settings['q0_treatment']
        inp.systems.sett.s1.bc1 = 'clamped'
        inp.engine = self.settings['engine']
        inp.driver.typeof = self.settings['driver']
        inp.driver.sol_path = self.data.output_folder + 'intrinsic/'
        inp.simulation.typeof = self.settings['sim_type']
        inp.systems.sett.s1.solution = self.settings['solution']
        inp.systems.sett.s1.solver_library = self.settings['solver_library']
        inp.systems.sett.s1.solver_function = self.settings['solver_function']
        inp.systems.sett.s1.solver_settings = dict(solver_name=self.settings['solver_name'],
                                                   rtol=self.settings['rtol'],
                                                   atol=self.settings['atol'],
                                                   max_steps=self.settings['max_steps'],
                                                   norm=self.settings['norm'],
                                                   kappa=self.settings['kappa'])

        # FEM Inputs
        inp.fem.connectivity = {name: None for name in self.component_names}  # TODO: replace with connectivity
        inp.fem.num_modes = self.num_modes
        inp.fem.X = jnp.array(self.x)
        inp.fem.component_vect = self.node_names
        inp.fem.fe_order = np.array(self.node_numbers)
        inp.fem.grid = None
        inp.fem.Ka = self.k_global
        inp.fem.Ma = self.m_global
        inp.fem.eig_names = None
        inp.fem.eigenvals = jnp.array(self.evals_struct)
        inp.fem.eigenvecs = jnp.array(self.evecs_struct)
        inp.fem.eig_type = "inputs"

        if self.settings['remove_symmetric']:
            num_nodes = (inp.fem.X.shape[0] + 1) // 2
            keep_nodes = slice(0, num_nodes)
            keep_dof = slice(0, inp.fem.Ma.shape[0] // 2)

            inp.fem.X = inp.fem.X[keep_nodes, :]
            eigen_to_keep = (np.abs(inp.fem.eigenvecs[2, :]) > 1e-10)

            inp.fem.eigenvals = inp.fem.eigenvals[eigen_to_keep]
            inp.fem.eigenvecs = inp.fem.eigenvecs[keep_dof, eigen_to_keep]

            inp.fem.Ka = inp.fem.Ka[keep_dof, keep_dof]
            inp.fem.Ma = inp.fem.Ma[keep_dof, keep_dof]

            inp.fem.connectivity = {"A": None}

            inp.fem.num_modes = int(inp.fem.num_modes / 2)
            inp.fem.component_vect = inp.fem.component_vect[keep_nodes]
            inp.fem.fe_order = inp.fem.fe_order[keep_nodes]

            pass

        return inp

    def statespace_structure(self) -> None:
        """
        Convert statespace system from discrete to continuous time and partition by input
        """

        # Remove structural states
        self.states_keep = []
        for i_s in range(self.data.linear.ss.state_variables.num_variables):
            if self.data.linear.ss.state_variables.vector_variables[i_s].name not in ['q', 'q_dot']:
                self.states_keep += list(self.data.linear.ss.state_variables.vector_variables[i_s].cols_loc)

        # Remove non-forcing outputs
        self.outputs_keep = []
        for i_s in range(self.data.linear.ss.output_variables.num_variables):
            if self.data.linear.ss.output_variables.vector_variables[i_s].name == 'Q':
                self.outputs_keep += list(self.data.linear.ss.output_variables.vector_variables[i_s].rows_loc)

        # Truncate states and outputs
        ss_d_trunc = pyyeti.ssmodel.SSModel(self.data.linear.ss.A[np.ix_(self.states_keep, self.states_keep)], \
                                            self.data.linear.ss.B[self.states_keep, :], \
                                            self.data.linear.ss.C[np.ix_(self.outputs_keep, self.states_keep)], \
                                            self.data.linear.ss.D[self.outputs_keep, :], \
                                            self.data.linear.ss.dt)

        # Convert to continuous time state space model
        self.ss_c_time = ss_d_trunc.d2c(self.settings['d2c_method'])

        # Split into three state space systems for each input
        # The A and C matrices are constant between the three

        self.ss_A = self.ss_c_time.A
        self.ss_C = self.ss_c_time.C
        self.ss_B0 = None
        self.ss_B1 = None
        self.ss_D0 = None
        self.ss_D1 = None

        self.num_lags = self.ss_A.shape[0]

        for i_s in range(self.data.linear.ss.input_variables.num_variables):
            var_name = self.data.linear.ss.input_variables.vector_variables[i_s].name
            param_index = self.data.linear.ss.input_variables.vector_variables[i_s].cols_loc
            match var_name:
                case 'q':
                    self.ss_B0 = self.ss_c_time.B[:, param_index]
                    self.ss_D0 = self.ss_c_time.D[:, param_index]
                case 'q_dot':
                    self.ss_B1 = self.ss_c_time.B[:, param_index]
                    self.ss_D1 = self.ss_c_time.D[:, param_index]
                case 'u_gust':
                    pass

        assert not (self.ss_B0 is None
                    or self.ss_B1 is None
                    or self.ss_D0 is None
                    or self.ss_D1 is None), \
            "Missing partition of state space inputs"

    def jig_loads(self) -> None:
        """
        Calculate loads in the jig shape in the material FoR, present due to twist or AoA
        """

        fm_total = self.data.aero.timestep_info[self.settings['use_custom_timestep']].forces

        self.fm_jig_nodal = np.zeros((6, self.data.structure.num_node))

        for i_surf in range(len(fm_total)):
            f_surf = fm_total[i_surf][:3, ...]  # 3 x M+1 x N+1
            f_node = np.sum(f_surf, 1)

            beam_pos = self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos
            zeta = self.data.aero.timestep_info[self.settings['use_custom_timestep']].zeta

            m, n = zeta[i_surf].shape[1:]

            m_node = np.zeros_like(f_node)
            for i_N in range(n):
                for i_M in range(m):
                    i_beam = self.data.aero.aero2struct_mapping[i_surf][i_N]

                    r = zeta[i_surf][:, i_M, i_N] - beam_pos[i_beam, :]
                    r_skew = algebra.skew(r)
                    m_node[:, i_N] += r_skew @ f_surf[:, i_M, i_N]

            rmat_ga = algebra.quat2rotation(self.settings['orientation'])
            f_node_g = rmat_ga @ f_node
            m_node_g = rmat_ga @ m_node

            self.fm_jig_nodal[:, self.data.aero.aero2struct_mapping[i_surf]] += np.vstack((f_node_g, m_node_g))

        if self.settings['remove_symmetric']:
            keep_nodes = slice(0, (self.data.structure.num_node + 1) // 2)
            self.fm_jig_modal = jnp.einsum('ijk, jk->i', self.phi1, self.fm_jig_nodal[:, keep_nodes])
        else:
            self.fm_jig_modal = jnp.einsum('ijk, jk->i', self.phi1, self.fm_jig_nodal)

    def calculate_eigs(self) -> [np.ndarray, np.ndarray]:
        """
        Structural eigendecomposition to give the square of the natural frequencies
        and the linear normal mode shapes, scaled to have same generalised coordinate as aeroelastic state space model
        """

        evecs_modal = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['eigenvectors']
        evals_global, evecs_global = np.linalg.eig(np.linalg.inv(self.m_global) @ self.k_global)
        i_order = np.argsort(evals_global)[:self.num_modes]
        evals_global = evals_global[i_order]
        evecs_global = evecs_global[:, i_order]

        end_indices = np.where(self.data.structure.boundary_conditions == -1)[0]  # array of all end index
        overall_scaling = np.zeros(self.num_modes)  # values per mode
        has_been_scaled = np.zeros(self.num_modes, dtype=bool)  # bool if mode is already scaled

        has_disp_global = np.zeros((end_indices.shape[0], self.num_modes), dtype=bool)
        has_disp_modal = np.zeros((end_indices.shape[0], self.num_modes), dtype=bool)
        for i_end, end in enumerate(end_indices):
            for i_mode in range(self.num_modes):
                has_disp_global[i_end, i_mode] = np.any(evecs_global[(end - 1) * 6:end * 6, i_mode])
                has_disp_modal[i_end, i_mode] = np.any(evecs_modal[(end - 1) * 6:end * 6, i_mode])

        # order modes the same in both cases
        evals_set = sorted(list(set(np.round(evals_global, 4))))
        new_order = np.zeros(self.num_modes, dtype=int)
        mode_count = 0
        for i_eval, eval in enumerate(evals_set):
            i_modes_modal = np.where(np.abs(evals_global - eval) < 1e-3)[0]

            match len(i_modes_modal):
                case 1:
                    new_order[mode_count] = mode_count
                case 2:
                    if np.all(has_disp_global[:, mode_count:mode_count + 2] == has_disp_modal[:,
                                                                               mode_count:mode_count + 2]):
                        new_order[mode_count:mode_count + 2] = [mode_count, mode_count + 1]
                    else:
                        new_order[mode_count:mode_count + 2] = [mode_count + 1, mode_count]
                case _:
                    raise ValueError
            mode_count += i_modes_modal.shape[0]
        evecs_global = evecs_global[:, new_order]

        # scale modes to keep displacements at the first displaced free end consistent
        for i_end in end_indices:
            slice_end = slice((i_end - 1) * 6, (i_end - 1) * 6 + 3)

            tip_disp_modal = np.linalg.norm(evecs_modal[slice_end, :], axis=0)
            tip_disp_global = np.linalg.norm(evecs_global[slice_end, :], axis=0)

            is_neg_m = np.sign(evecs_modal[slice_end, :][
                                   np.argmax(np.abs(evecs_modal[slice_end, :]), axis=0), np.arange(self.num_modes)])
            is_neg_g = np.sign(evecs_global[slice_end, :][
                                   np.argmax(np.abs(evecs_global[slice_end, :]), axis=0), np.arange(self.num_modes)])

            scaling = tip_disp_modal / tip_disp_global * is_neg_g * is_neg_m

            is_valid_scaling = ~np.isnan(scaling) & (scaling != 0.0)
            overall_scaling += np.nan_to_num(
                scaling * is_valid_scaling * ~has_been_scaled)  # scale modes if values are valid and isnt already scaled
            has_been_scaled |= is_valid_scaling  # update mask of scaled modes

            if np.all(has_been_scaled):
                break

        evecs_scaled = evecs_global @ np.diag(overall_scaling)
        return evals_global, evecs_scaled

    def find_static_sol(self) -> np.array:
        """
        Find the static solution to the system
        """

        gamma2 = jnp.array(self.gamma2)
        eta_jig = jnp.array(self.fm_jig_modal)

        e_mat = jnp.real(
            jnp.array(np.diag(self.omega) + (self.ss_C @ np.linalg.inv(self.ss_A) @ self.ss_B0 - self.ss_D0)
                      @ np.diag(1.0 / self.omega)))

        def f(q2: jnp.ndarray):
            return e_mat @ q2 - jnp.einsum('jik,i,k->j', gamma2, q2, q2) + eta_jig

        def f_prime_inv(q2: jnp.ndarray):
            return jnp.linalg.inv(jax.jacfwd(f, argnums=0)(q2))

        q2 = -jnp.ones_like(self.fm_jig_modal)

        # iterate
        for i_iter in range(self.settings['max_iter']):
            f_val = f(q2)
            res = np.linalg.norm(f_val)
            if res < 1e-9:
                break
            elif i_iter == self.settings['max_iter'] - 1:
                raise RuntimeError(f"Static solution not converged, Residual: {res:.2f}")
            f_prime_inv_val = f_prime_inv(q2)
            q2 = q2 - f_prime_inv_val @ f_val

        self.q2_bar = np.array(q2)
        self.q0_bar = -np.diag(1.0 / self.omega) @ self.q2_bar
        self.lambda_bar = -np.linalg.inv(self.ss_A) @ self.ss_B0 @ np.diag(1.0 / self.omega) @ self.q2_bar

        # self.input.systems.sett.s1.init_states = dict(q2=["prescribed", jnp.array(self.q2_bar)])
        if self.settings['remove_symmetric']:
            self.input.systems.sett.s1.q0_input = np.concatenate((np.zeros(self.num_modes // 2), self.q2_bar))
        else:
            self.input.systems.sett.s1.q0_input = np.concatenate((np.zeros(self.num_modes), self.q2_bar))
        self.input.systems.sett.s1.t1 = 1e-8

        config = Config(self.input)
        sol = fem4inas_main.main(input_obj=config)

        return np.array(sol.dynamicsystem_s1.ra[0, ...])

    def assemble_sys(self) -> None:

        elem12 = ((np.diag(self.omega) - np.einsum('jik,k->ji', self.gamma2, self.q2_bar)
                   - np.einsum('jik,i->jk', self.gamma2, self.q2_bar))
                  - self.ss_D0 @ np.diag(1.0 / self.omega))

        elem21 = -np.diag(self.omega) + np.einsum('ijk,k->ji', self.gamma2, self.q2_bar)
        elem32 = -self.ss_B0 @ np.diag(1.0 / self.omega)

        if self.settings['remove_symmetric']:
            self.sys_mat = np.block([[self.ss_D1, elem12, self.ss_C],
                                     [elem21, np.zeros((self.num_modes // 2, self.num_modes // 2 + self.num_lags))],
                                     [self.ss_B1, elem32, self.ss_A]])
        else:
            self.sys_mat = np.block([[self.ss_D1, elem12, self.ss_C],
                                 [elem21, np.zeros((self.num_modes, self.num_modes + self.num_lags))],
                                 [self.ss_B1, elem32, self.ss_A]])

        self.evals_ae, self.evecs_ae = np.linalg.eig(self.sys_mat)

        cout.cout_wrap("Validating system stability", 0)

        is_stable = self.evals_ae.real < 0.0
        cout.cout_wrap(f"Stable: {np.all(is_stable)}", 1)
        if not np.all(is_stable):
            cout.cout_wrap(f"Unstable Eigenvalues:", 1)
            unstable_evals = self.evals_ae[~is_stable]
            for eval in unstable_evals:
                cout.cout_wrap(str(eval), 2)
            cout.cout_wrap(f"Unstable Frequencies:", 1)
            for eval in unstable_evals:
                cout.cout_wrap(str(np.abs(eval) / (2.0 * np.pi)) + ' Hz', 2)
