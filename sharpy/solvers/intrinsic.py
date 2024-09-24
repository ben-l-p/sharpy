# Python package imports
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
class IntrinsicSolver(BaseSolver):
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
    solver_id = 'Intrinsic'
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
    settings_default['aero_approx'] = 'roger'
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

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = 0.0
    settings_description['gust_intensity'] = 'Intensity of gust (m/s)'

    settings_types['gust_length'] = 'float'
    settings_default['gust_length'] = 0.0
    settings_description['gust_length'] = 'Length of gust'

    settings_types['gust_num_x'] = 'int'
    settings_default['gust_num_x'] = 0
    settings_description['gust_num_x'] = ('Number of spatial points for the gust to be evaluated, '
                                          'between which they are interpolated')

    settings_types['gust_offset'] = 'float'
    settings_default['gust_offset'] = 0.0
    settings_description['gust_offset'] = 'Offset distance of gust'

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

    settings_types['t1'] = 'float'
    settings_default['t1'] = None
    settings_description['t1'] = 'Simulation time (s)'

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Number of simulation steps'

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

    settings_types['aero_on'] = 'bool'
    settings_default['aero_on'] = True
    settings_description['aero_on'] = 'Enable aerodynamics'

    settings_types['gust_on'] = 'bool'
    settings_default['gust_on'] = False
    settings_description['gust_on'] = 'Enable gusts'

    settings_types['q0_treatment'] = 'int'
    settings_default['q0_treatment'] = 2
    settings_description['q0_treatment'] = 'Method for obtaining q0'
    #
    settings_types['tip_force'] = 'list(float)'
    settings_default['tip_force'] = [0., 0., 0.]
    settings_description['tip_force'] = 'Add tip forcing to the final node (for testing)'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = Optional[sharpy.presharpy.presharpy.PreSharpy]
        self.settings: Optional[dict] = None
        self.m_global: Optional[np.ndarray] = None
        self.k_global: Optional[np.ndarray] = None
        self.tn: Optional[int] = None
        self.aero_model: Optional[str] = None
        self.gust_on: Optional[bool] = None
        self.aero_on: Optional[bool] = None
        self.x: Optional[np.ndarray] = None
        self.conn: Optional[np.ndarray] = None
        self.beam_number: Optional[np.ndarray] = None
        self.num_modes: Optional[int] = None
        self.num_nodes: Optional[int] = None
        self.node_numbers: Optional[list[int]] = None
        self.component_names: Optional[list[str]] = None
        self.node_names: Optional[list[str]] = None

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
            self.r_a = np.squeeze(algebra.quat2rot(orientation) @ np.expand_dims(self.r_g, -1))
            self.t = np.array(sol.dynamicsystem_s1.t)

            try:
                self.eta_a_jig = np.array(
                    sol.modalaerostatespace_s1.eta_a_jig)  # TODO: make this work for other aero models
                self.f_jig = np.array(sol.modalaerostatespace_s1.f_jig)
            except AttributeError:
                pass

            self.gamma1 = np.array(sol.couplings.gamma1)
            self.gamma2 = np.array(sol.couplings.gamma2)

            # self.phi1 = np.array(sol.modes.phi1)
            # self.psi1 = np.array(sol.modes.psi1l)
            # self.phi2 = np.array(sol.modes.phi2)
            # self.psi2 = np.array(sol.modes.psi2l)

            self.modes = sol.modes

    def run(self, **kwargs) -> sharpy.presharpy.presharpy.PreSharpy:
        self.tn = int(self.settings['t1'] / self.settings['dt'])

        # Create all case inputs
        self.get_grid()
        self.transform_struct()
        self.evals, self.evecs = self.calculate_eigs()
        self.jig_loads()

        # Add aero attributes from input aero model
        self.aero_model: str = self.settings['aero_approx']
        self.gust_on: bool = self.settings['gust_on']
        self.aero_on: bool = self.settings['aero_on']
        if self.aero_on:
            if self.aero_model == 'roger':
                self.roger_structure()
                if self.gust_on:
                    self.roger_gust()
            elif self.aero_model == 'statespace':
                self.statespace_structure()
            else:
                raise KeyError(f"Aero model {self.aero_model} not recognised")

        cout.cout_wrap(f"Number of time steps: {self.tn}", 0)

        # Check system eigenvalues
        self.check_stability()

        # Generate case object
        input = self.generate_settings_file()
        config = Config(input)

        # Run case
        cout.cout_wrap("\nRunning Intrinsic Solver", 0)
        sol = fem4inas_main.main(input_obj=config)
        cout.cout_wrap("Intrinsic Solution Complete", 0)

        if jnp.any(jnp.isnan(sol.dynamicsystem_s1.q)) or jnp.any(jnp.isinf(sol.dynamicsystem_s1.q)):
            cout.cout_wrap("\tWarning - output contains NaN or Inf values", 1)

        intrinsic_out = self.IntrinsicObj(sol, self.m_global, self.k_global, self.settings['orientation'])
        self.data.intrinsic = intrinsic_out

        return self.data

    # Returns the grid of the system
    def get_grid(self) -> None:
        # Create component names
        n_beams = self.data.structure.num_bodies
        if self.settings['component_names']:  # TODO: fix the issue with settings passed as strings
            assert n_beams == len(
                self.settings['component_names']), "Number of component names does not match number of components"
            self.component_names = self.settings['component_names']
        else:
            self.component_names = [chr(65 + i) for i in range(n_beams)]

        rot_orient = algebra.quat2rot(self.settings['orientation']).T
        # Create grid
        self.x = np.squeeze(rot_orient @ np.expand_dims(
            self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos, -1))
        self.conn = self.data.structure.connectivities
        self.beam_number = self.data.structure.beam_number

        self.num_modes = self.settings['num_modes']
        self.node_numbers = list(range(-1, self.data.structure.num_node - 1))  #TODO - make this use SHARPy node numbers
        self.node_names = [self.component_names[0]] + [self.component_names[i] for i in self.beam_number for _ in
                                                       (0, 1)]

    def transform_struct(self) -> None:
        # transform mass and stiffness to global frame
        m_modal = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['M']
        k_modal = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['K']

        num_nodes = m_modal.shape[0] // 6

        psi0 = np.vstack((self.data.structure.timestep_info[self.settings['use_custom_timestep']].psi[0, 0, :],
                          self.data.structure.timestep_info[self.settings['use_custom_timestep']].psi[:, [2, 1], :]
                          .reshape(-1, 3)))

        tan_psi0 = np.apply_along_axis(algebra.crv2tan, axis=1, arr=psi0)

        rot_orient = algebra.quat2rot(self.settings['orientation'])

        tfrm = np.linalg.inv(block_diag(
            *[np.block([[rot_orient, np.zeros((3, 3))], [np.zeros((3, 3)), rot_orient @ tan_psi0[i + 1, ...]]])
              for i in range(num_nodes)]))

        i_tfrm = tfrm.T

        self.m_global = tfrm @ m_modal @ i_tfrm
        self.k_global = tfrm @ k_modal @ i_tfrm

    # Return collocation points at the leading edge of all surfaces
    def calculate_collocation_dihedral(self):
        col = []

        ts_data = self.data.aero.timestep_info[self.settings['use_custom_timestep']]
        for surf in ts_data.zeta:
            for i_N in range(surf.shape[2]):
                for i_M in range(surf.shape[1]):
                    # TODO: optimise this code and add dihedral (this is already rotated in linear UVLM!)
                    col.append([surf[0, i_M, i_N], surf[1, i_M, i_N], surf[2, i_M, i_N]])

        dih = np.ones_like(col)
        return np.array(col), np.array(dih)

    def generate_settings_file(self) -> Inputs:
        """
        Generate case file which is passed to FENIAX
        """

        inp = Inputs()

        # General settings
        inp.systems.sett.s1.t1 = self.settings['t1']
        inp.systems.sett.s1.tn = self.tn
        inp.systems.sett.s1.aero.rho_inf = self.settings['rho']
        inp.systems.sett.s1.aero.u_inf = self.settings['u_inf']
        inp.systems.sett.s1.aero.c_ref = self.settings['c_ref']
        inp.systems.sett.s1.xloads.modalaero_forces = self.aero_on
        inp.systems.sett.s1.xloads.gravity_forces = self.settings['gravity_on']
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
        inp.fem.connectivity = dict(A=None)  #TODO: replace with connectivity
        inp.fem.num_modes = self.num_modes
        inp.fem.X = jnp.array(self.x)
        inp.fem.component_vect = self.node_names
        inp.fem.fe_order = np.array(self.node_numbers)
        inp.fem.grid = None
        inp.fem.Ka = self.k_global
        inp.fem.Ma = self.m_global
        inp.fem.eig_names = None
        inp.fem.eigenvals = jnp.array(self.evals)
        inp.fem.eigenvecs = jnp.array(self.evecs)
        inp.fem.eig_type = "inputs"

        # Tip forcing
        if np.any(self.settings['tip_force']):
            inp.systems.sett.s1.xloads.follower_forces = True
            inp.systems.sett.s1.xloads.follower_points = [[self.x.shape[0]-1, 0],
                                                          [self.x.shape[0]-1, 1],
                                                          [self.x.shape[0]-1, 2]]
            inp.systems.sett.s1.xloads.x = [0, 1e5]

            tip_force = self.settings['tip_force']
            inp.systems.sett.s1.xloads.follower_interpolation = [[float(tip_force[0]), float(tip_force[0])],
                                                                 [float(tip_force[1]), float(tip_force[1])],
                                                                 [float(tip_force[2]), float(tip_force[2])]]

        # inp.systems.sett.s1.xloads.force_follower = jnp.array(load_follower)

        # Structural aero inputs
        if self.aero_on:
            match self.aero_model:
                case 'roger':
                    inp.systems.sett.s1.aero.poles = -jnp.array(self.roger_poles)
                    inp.systems.sett.s1.aero.A = jnp.array(self.roger_A)

                case 'statespace':
                    inp.systems.sett.s1.aero.approx = 'statespace'
                    inp.systems.sett.s1.aero.ss_A = jnp.array(self.ss_A, dtype=float)
                    inp.systems.sett.s1.aero.ss_B0 = jnp.array(self.ss_B0, dtype=float)
                    inp.systems.sett.s1.aero.ss_B1 = jnp.array(self.ss_B1, dtype=float)
                    inp.systems.sett.s1.aero.ss_C = jnp.array(self.ss_C, dtype=float)
                    inp.systems.sett.s1.aero.ss_D0 = jnp.array(self.ss_D0, dtype=float)
                    inp.systems.sett.s1.aero.ss_D1 = jnp.array(self.ss_D1, dtype=float)
                    inp.systems.sett.s1.aero.f_jig = jnp.array(self.fm_jig_nodal, dtype=float)

                case _:
                    raise KeyError(f"Model {self.aero_model} is not recognised")

        # Gust aero inputs
        if self.aero_on and self.gust_on:
            col, dih = self.calculate_collocation_dihedral()
            inp.systems.sett.s1.aero.gust.panels_dihedral = jnp.array(dih)
            inp.systems.sett.s1.aero.gust.collocation_points = jnp.array(col)
            inp.systems.sett.s1.aero.gust_profile = "mc"
            inp.systems.sett.s1.aero.gust.intensity = self.settings['gust_intensity']
            inp.systems.sett.s1.aero.gust.length = self.settings['gust_length']
            inp.systems.sett.s1.aero.gust.step = self.settings['gust_length'] / self.settings['gust_num_x']
            inp.systems.sett.s1.aero.gust.shift = self.settings['gust_offset']

            match self.aero_model:
                case 'roger':
                    inp.systems.sett.s1.aero.D = jnp.array(self.roger_D)
                case 'statespace':
                    assert self.ss_Bw is not None and self.ss_Dw is not None, "No gust matrices in statespace"
                    inp.systems.sett.s1.aero.ss_Bw = jnp.array(self.ss_Bw[:, 2::3], dtype=float)
                    inp.systems.sett.s1.aero.ss_Dw = jnp.array(self.ss_Dw[:, 2::3], dtype=float)

        return inp

    def roger_structure(self) -> None:
        """
        Convert matrix for Roger structural aero
        """

        if not hasattr(self.data.linear, 'rfa'):
            raise AttributeError("RFA postproccesor needs to be run")

        # Aero due to structure
        self.roger_poles = self.data.linear.rfa.poles

        self.roger_A = np.zeros([3 + len(self.roger_poles), self.num_modes, self.num_modes], dtype=float)
        self.roger_A[0, :, :] = self.data.linear.rfa.matrices_q[0]
        self.roger_A[1, :, :] = self.data.linear.rfa.matrices_q[1]
        for i_mat in range(len(self.data.linear.rfa.poles)):
            self.roger_A[i_mat + 3, :, :] = self.data.linear.rfa.matrices_q[i_mat + 2]

    def roger_gust(self) -> None:
        """
        Convert matrix for Roger gust
        """

        # Check RFA exists
        if self.data.linear.rfa.matrices_w is None:
            raise AttributeError("Gust matrices need to be generated by RFA postprocessor")

        # Find index of leading edge elements in state space gust vector
        leading_edge_index = []
        prev_end = 0
        for surf_dim in self.data.aero.timestep_info[self.settings['use_custom_timestep']].dimensions:
            leading_edge_index.extend((np.arange(surf_dim[1] + 1) * (surf_dim[0] + 1)) + prev_end)
            prev_end += (surf_dim[0] + 1) * (surf_dim[1] + 1)

        # Create gust matrices for leading edge panels only
        n_w = len(leading_edge_index)
        self.roger_D = np.zeros([3 + len(self.data.linear.rfa.poles), self.num_modes, n_w], dtype=float)
        self.roger_D[0, :, :] = self.data.linear.rfa.matrices_w[0][:, [3 * i for i in leading_edge_index]]
        for i_mat in range(len(self.data.linear.rfa.poles)):
            self.roger_D[i_mat + 3, :, :] = self.data.linear.rfa.matrices_w[i_mat + 1][:,
                                            [3 * i for i in leading_edge_index]]

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
        self.ss_c = ss_d_trunc.d2c(self.settings['d2c_method'])

        # Split into three state space systems for each input
        # The A and C matrices are constant between the three

        self.ss_A = self.ss_c.A
        self.ss_C = self.ss_c.C
        self.ss_B0 = None
        self.ss_B1 = None
        self.ss_D0 = None
        self.ss_D1 = None
        self.ss_Bw = None
        self.ss_Dw = None

        self.num_lags = self.ss_A.shape[0]

        for i_s in range(self.data.linear.ss.input_variables.num_variables):
            var_name = self.data.linear.ss.input_variables.vector_variables[i_s].name
            param_index = self.data.linear.ss.input_variables.vector_variables[i_s].cols_loc
            match var_name:
                case 'q':
                    self.ss_B0 = self.ss_c.B[:, param_index]
                    self.ss_D0 = self.ss_c.D[:, param_index]
                case 'q_dot':
                    self.ss_B1 = self.ss_c.B[:, param_index]
                    self.ss_D1 = self.ss_c.D[:, param_index]
                case 'u_gust':
                    self.ss_Bw = self.ss_c.B[:, param_index]
                    self.ss_Dw = self.ss_c.D[:, param_index]

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

        for i_surf in range(len(fm_total)):  # TODO: support multiple surfaces
            f_surf = fm_total[i_surf][:3, ...]  # 3 x M+1 x N+1
            f_node = np.sum(f_surf, 1)

            beam_pos = self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos
            zeta = self.data.aero.timestep_info[self.settings['use_custom_timestep']].zeta

            m, n = zeta[i_surf].shape[1:]

            m_node = np.zeros_like(f_node)
            for i_N in range(n):
                for i_M in range(m):
                    r = zeta[i_surf][:, i_M, i_N] - beam_pos[i_N, :]
                    r_skew = algebra.skew(r)
                    m_node[:, i_N] += r_skew @ f_surf[:, i_M, i_N]

            self.fm_jig_nodal = np.vstack((f_node, m_node))

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

        tip_disp_modal = np.linalg.norm(evecs_modal[-6:-3, :], axis=0)
        tip_disp_global = np.linalg.norm(evecs_global[-6:-3, :], axis=0)

        elem_size_m = evecs_modal[-6:-3, :][np.argmax(np.abs(evecs_modal[-6:-3, :]), axis=0), np.arange(self.num_modes)]
        is_neg_m = np.array([-1. if e < 0. else 1. for e in elem_size_m])

        elem_size_g = evecs_global[-6:-3, :][np.argmax(np.abs(evecs_global[-6:-3, :]), axis=0), np.arange(self.num_modes)]
        is_neg_g = np.array([-1. if e < 0. else 1. for e in elem_size_g])

        scaling = tip_disp_modal / tip_disp_global * is_neg_g * is_neg_m

        evecs_scaled = evecs_global @ np.diag(scaling)
        return evals_global, evecs_scaled

    def check_stability(self) -> None:
        # """
        # Assess the stability of the contructed aeroelastic system
        # """

        r"""Stability in continuous time can be checked through construction of aeroelastic statespace form,
        with the state matrix defined as \begin{equation} \mathbf{A}_{AE} = \begin{bmatrix} \mathbf{D}_{1} & \mathbf{
        \Omega} - \mathbf{D}_0 \mathbf{\Omega}^{-1} & \mathbf{C} \\ -\mathbf{\Omega} & \mathbf{0} & \mathbf{0} \\
        \mathbf{B}_1 & -\mathbf{B}_0 \mathbf{\Omega}^{-1} & \mathbf{A} \end{bmatrix} \end{equation} for the
        aerodynamic system $\mathbf{\Sigma}_A = \{\mathbf{A, B, C, D}\}$ and system states $\textbf{x}_{AE} = \{
        \mathbf{q}_1; \mathbf{q}_2; \mathbf{x}_A\}$ to give $\dot{\mathbf{x}}_{AE} = \mathbf{A}_{AE} \mathbf{x}_{
        AE}$. The stability can be verified with analysis of the eigenvalues of $\mathbf{A}_{AE}$. This uses the
        identity $\mathbf{q}_0 = \mathbf{q}_2 \mathbf{\Omega}^{-1}$."""

        if not self.aero_on:
            cout.cout_wrap("Stability analysis can only be performed with aerodynamics enabled - skipping", 0)
            return

        if self.aero_model != 'statespace':
            cout.cout_wrap("Stability analysis can only be performed with statespace aero - skipping", 0)
            return

        evals, evects = np.linalg.eig(np.linalg.inv(self.m_global) @ self.k_global)

        evals = np.sort(evals)[:self.num_modes]

        # Check stability of original aeroelastic state space
        cout.cout_wrap("Calculating discrete time aeroelastic state space stability", 0)
        evals_oae, _ = np.linalg.eig(self.data.linear.ss.A)
        unstable_i_aed = np.where(np.abs(evals_oae) > 1.0, 1, 0)
        if np.any(unstable_i_aed):
            cout.cout_wrap("\tSystem unstable", 1)
        else:
            cout.cout_wrap("\tSystem stable", 1)

        # Check aero stability
        cout.cout_wrap("Calculating continuous time aerodynamic state space stability", 0)
        evals_a, _ = np.linalg.eig(self.ss_A)
        if np.any(np.where(np.real(evals_a) > 1e-5, 1, 0)):
            cout.cout_wrap("\tSystem unstable", 1)
        else:
            cout.cout_wrap("\tSystem stable", 1)

        # Create continuous time state matrix
        cout.cout_wrap("Calculating continuous time linear aeroelastic state space stability", 0)
        omega = np.diag(np.sqrt(evals))
        iomega = np.diag(1. / np.sqrt(evals))
        a_ae = np.vstack((
            np.hstack((self.ss_D1, omega - self.ss_D0 @ iomega, self.ss_C)),
            np.hstack((-omega, np.zeros((self.num_modes, self.num_modes + self.num_lags)))),
            np.hstack((self.ss_B1, -self.ss_B0 @ iomega, self.ss_A))))

        # Check stability of continuous time aeroelastic system
        evals_ae, _ = np.linalg.eig(a_ae)
        unstable_i_aec = np.where(np.real(evals_ae) > 1e-5, 1, 0)
        if np.any(unstable_i_aec):
            cout.cout_wrap("\tSystem unstable", 1)
            evals_unstable_ae = np.extract(unstable_i_aec, evals_ae)
            cout.cout_wrap(f"\tNumber of unstable eigenvalues: {len(evals_unstable_ae)}", 1)
            for eval in evals_unstable_ae:
                cout.cout_wrap(f"\t\t{np.real(eval):.3f} + {np.imag(eval):.3f}i", 2)
        else:
            cout.cout_wrap("\tSystem stable", 1)

        if np.any(unstable_i_aec) ^ np.any(unstable_i_aed):
            warnings.warn("Continuous time aeroelastic system does not "
                          "preserve stability characteristics of discrete time system")
