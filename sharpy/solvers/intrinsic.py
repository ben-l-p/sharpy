# Python package imports
import numpy as np
import jax.numpy as jnp
import pyyeti
import typing

# General SHARPy imports
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.cout_utils as cout
import sharpy.presharpy.presharpy

# FEM4INAS
from fem4inas import fem4inas_main
from fem4inas.preprocessor.configuration import Config
from fem4inas.preprocessor.inputs import Inputs

@solver
class IntrinsicSolver(BaseSolver):
    """
    Description of intrinsic solver...
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

    settings_types['aero_approx'] = 'str'
    settings_default['aero_approx'] = 'roger'
    settings_description['aero_approx'] = 'Aerodynamic model to use.'
    settings_options['aero_approx'] = ['roger', 'statespace', 'none']

    settings_types['delta_curved'] = 'float'
    settings_default['delta_curved'] = 1e-2
    settings_description['delta_curved'] = 'Threshold for linear rotations'

    settings_types['plot_eigenvalues'] = 'bool'
    settings_default['plot_eigenvalues'] = False
    settings_description['plot_eigenvalues'] = 'Plot to screen root locus diagram'

    settings_types['use_custom_timestep'] = 'int'
    settings_default['use_custom_timestep'] = 0
    settings_description['use_custom_timestep'] = 'Time step of structure for calculating modes'

    settings_types['component_names'] = 'list(str)'
    settings_default['component_names'] = []
    settings_description['component_names'] = 'Name components of the structure. Will use lettering [A, B, ...] by default'

    settings_types['dynamic_tstep_init'] = 'int'
    settings_default['dynamic_tstep_init'] = -1
    settings_description['dynamic_tstep_init'] = 'Time step of structure for calculating q0 - doesnt work yet'

    settings_types['stability_analysis'] = 'bool'
    settings_default['stability_analysis'] = False
    settings_description['stability_analysis'] = "Sweep through velocitities to determine if solution is stable"

    settings_types['stability_v_min'] = 'float'
    settings_default['stability_v_min'] = None
    settings_description['stability_v_min'] = 'Minimum velocity for stability analysis'

    settings_types['stability_v_max'] = 'float'
    settings_default['stability_v_max'] = None
    settings_description['stability_v_max'] = 'Maximum velocity for stability analysis'

    settings_types['stability_v_num'] = 'int'
    settings_default['stability_v_num'] = None
    settings_description['stability_v_num'] = 'Number of velocities for stability analysis'

    settings_types['d2c_method'] = 'str'
    settings_default['d2c_method'] = 'foh'
    settings_description ['d2c_method'] = 'Method for converting state space from discrete to continuous time'
    settings_options['d2c_method'] = ['zoh', 'zoha', 'foh', 'tustin']

    settings_types['gust_intensity'] = 'float'
    settings_default['gust_intensity'] = 0.0
    settings_description['gust_intensity'] = 'yes'

    settings_types['gust_length'] = 'float'
    settings_default['gust_length'] = 0.0
    settings_description['gust_length'] = 'yes'

    settings_types['gust_num_x'] = 'int'
    settings_default['gust_num_x'] = 0
    settings_description['gust_num_x'] = 'yes'

    settings_types['gust_offset'] = 'float'
    settings_default['gust_offset'] = 0.0
    settings_description['gust_offset'] = 'yes'

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

    settings_types['tn'] = 'int'
    settings_default['tn'] = 10000
    settings_description['tn'] = 'Number of simulation steps'

    settings_types['t_factor'] = 'float'
    settings_default['t_factor'] = -1.0
    settings_description['t_factor'] = 'Multiple of Nyquist step size for highest frequency mode to use. Set to -1 to be disabled and set number of steps using tn'

    settings_types['rho'] = 'float'
    settings_default['rho'] = None
    settings_description['rho'] = 'Freestream density (kg/m^3)'

    settings_types['u_inf'] = 'float'
    settings_default['u_inf'] = None
    settings_description['u_inf'] = 'Freestream velocity (m/s)'

    settings_types['c_ref'] = 'float'
    settings_default['c_ref'] = None
    settings_description['c_ref'] = 'Reference chord (m)'

    settings_types['reduced_time_ss'] = 'bool'
    settings_default['reduced_time_ss'] = True
    settings_description['reduced_time_ss'] = "True for state space system in terms of reduced time, false for in terms of real time"

    settings_types['gravity_on'] = 'bool'
    settings_default['gravity_on'] = True
    settings_description['gravity_on'] = 'Enable gravity'

    settings_types['aero_on'] = 'bool'
    settings_default['aero_on'] = True
    settings_description['aero_on'] = 'Enable aerodynamics'

    settings_types['gust_on'] = 'bool'
    settings_default['gust_on'] = False
    settings_description['gust_on'] = 'Enable gusts'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data: sharpy.presharpy.presharpy.PreSharpy, custom_settings=None, restart=False):
        # Load solver settings
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)	

    class Intrinsic_Obj:
        def __init__(self):
            self.Cab = None
            self.X1 = None
            self.X2 = None
            self.X3 = None
            self.q = None
            self.ra = None
            self.t = None
            self.stability_analysis = None

        def update_sol(self, sol):
            self.Cab = np.array(sol.dynamicsystem_s1.Cab)
            self.X1 = np.array(sol.dynamicsystem_s1.X1)
            self.X2 = np.array(sol.dynamicsystem_s1.X2)
            self.X3 = np.array(sol.dynamicsystem_s1.X3)
            self.q = np.array(sol.dynamicsystem_s1.q)
            self.ra = np.array(sol.dynamicsystem_s1.ra)
            self.t = np.array(sol.dynamicsystem_s1.t)

    def run(self, **kwargs) -> sharpy.presharpy.presharpy.PreSharpy:
        intrinsic_out = self.Intrinsic_Obj()
        # self.get_M_C_K(self.settings['use_custom_timestep'])
        # self.get_eigs0()
        self.get_grid()

        # Stability analysis
        if self.settings['stability_analysis']:
            cout.cout_wrap("Running stability analysis", 0)
            stab_out = []
            vels = np.linspace(self.settings['stability_v_min'],
                               self.settings['stability_v_max'], 
                               self.settings['stability_v_num'])
            for vel in vels:
                input = self.generate_settings_file(float(vel))
                config = Config(input)
                cout.cout_wrap(f"\nVelocity: {vel:.2f} m/s", 1)
                cout.cout_wrap("Running Intrinsic Solver", 0)
                sol = fem4inas_main.main(input_obj=config)
                cout.cout_wrap("Intrinsic Solution Complete", 0)

                if jnp.any(jnp.isnan(sol.dynamicsystem_s1.q)) or jnp.any(jnp.isinf(sol.dynamicsystem_s1.q)):
                    stab_out.append({'u_inf': vel, 'is_stable': False})
                    cout.cout_wrap("    Stable: False", 1)
                else:
                    stab_out.append({'u_inf': vel, 'is_stable': True})
                    cout.cout_wrap("    Stable: True", 1)

        # Case to save
        input = self.generate_settings_file(self.settings['u_inf'])
        config = Config(input)
        cout.cout_wrap("\nRunning Intrinsic Solver", 0)
        sol = fem4inas_main.main(input_obj=config)
        cout.cout_wrap("Intrinsic Solution Complete", 0)

        if jnp.any(jnp.isnan(sol.dynamicsystem_s1.q)) or jnp.any(jnp.isinf(sol.dynamicsystem_s1.q)):
            cout.cout_wrap("    Warning - model is unstable", 1)

        intrinsic_out.update_sol(sol)
        self.data.intrinsic = intrinsic_out


        return self.data

    # Returns the grid of the system
    def get_grid(self) -> None:
        # Create component names
        n_beams = self.data.structure.num_bodies
        if self.settings['component_names']:      # TODO: fix the issue with settings passed as strings
            assert n_beams == len(self.settings['component_names']), "Number of component names does not match number of components"
            self.component_names = self.settings['component_names']
        else:
            self.component_names = [chr(65+i) for i in range(n_beams)]

        # Create grid
        self.X = self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos

        # self.X = np.array([self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos[:, 1],
        #             self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos[:, 0],
        #             self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos[:, 2]]).T

        self.conn = self.data.structure.connectivities
        self.beam_number = self.data.structure.beam_number

        self.node_numbers = range(-1, self.data.structure.num_node-1)         #TODO - make this use SHARPy node numbers
        self.node_names = [self.component_names[0]] + [self.component_names[i] for i in self.beam_number for j in (0, 1)]

    # Return collocation points at the leading edge of all surfaces
    def calculate_collocation_dihedral(self):
        col = []
        dih = []

        ts_data = self.data.aero.timestep_info[self.settings['use_custom_timestep']]
        for surf in ts_data.zeta:
            for i_N in range(surf.shape[2]):
                for i_M in range(surf.shape[1]):
                    col.append([surf[0, i_M, i_N], surf[1, i_M, i_N], surf[2, i_M, i_N]])
                    dih.append([1.0,])      # TODO: optimise this code and add dihedral

        return np.array(col), np.array(dih)

    def generate_settings_file(self, u_inf) -> Inputs:
        inp = Inputs()

        # General settings
        inp.systems.sett.s1.t1 = self.settings['t1']
        inp.systems.sett.s1.aero.rho_inf = self.settings['rho']
        inp.systems.sett.s1.aero.u_inf = u_inf
        inp.systems.sett.s1.aero.c_ref = self.settings['c_ref']
        inp.systems.sett.s1.xloads.modalaero_forces = self.settings['aero_on']
        inp.systems.sett.s1.xloads.gravity_forces = self.settings['gravity_on']
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
                                                    kappa=self.settings['kappa'],
                                                    stepsize_controller=dict(PIDController=dict(atol=1e-5,rtol=1e-5)))

        # FEM Inputs
        inp.fem.connectivity = dict(A=None)             #TODO: replace with connectivity

        M = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['M']
        K = self.data.structure.timestep_info[self.settings['use_custom_timestep']].modal['K']

        inp.fem.Ka = jnp.array(K)
        inp.fem.Ma = jnp.array(M)
        
        inp.fem.num_modes = self.settings['num_modes']

        # inp.fem.eig_type = "inputs"
        inp.fem.X = jnp.array(self.X)
        inp.fem.component_vect = self.node_names
        inp.fem.fe_order = self.node_numbers
        inp.fem.grid = None
        inp.fem.eig_names = None

        # Roger aero inputs
        if self.settings['aero_approx'] == 'roger' and self.settings['aero_on']:
            if not hasattr(self.data.linear, 'rfa'):
                raise AttributeError("RFA postproccesor needs to be run")
            
            # Aero due to structure
            A = np.zeros([3 + len(self.data.linear.rfa.poles), self.settings['num_modes'], self.settings['num_modes']], dtype=float)
            A[0, :, :] = self.data.linear.rfa.matrices_q[0]
            A[1, :, :] = self.data.linear.rfa.matrices_q[1]
            for i_mat in range(len(self.data.linear.rfa.poles)):
                A[i_mat+3, :, :] = self.data.linear.rfa.matrices_q[i_mat+2]

            inp.systems.sett.s1.aero.poles = -jnp.array(self.data.linear.rfa.poles)
            inp.systems.sett.s1.aero.A = jnp.array(A)

            # Aero due to disturbances
            if self.settings['gust_on']:

                # Check RFA exists
                if self.data.linear.rfa.matrices_w is None:
                    raise AttributeError("Gust matrices need to be generated by RFA postprocessor")

                # Find index of leading edge elements in state space gust vector
                leading_edge_index = []
                prev_end = 0
                for surf_dim in self.data.aero.timestep_info[self.settings['use_custom_timestep']].dimensions:
                    leading_edge_index.extend((np.arange(surf_dim[1]+1)*(surf_dim[0]+1))+prev_end)
                    prev_end += (surf_dim[0]+1)*(surf_dim[1]+1)

                # Create gust matrices for leading edge panels only
                n_w = len(leading_edge_index)
                D = np.zeros([3 + len(self.data.linear.rfa.poles), self.settings['num_modes'], n_w], dtype=float)
                D[0, :, :] = self.data.linear.rfa.matrices_w[0][:, [3*i for i in leading_edge_index]]
                for i_mat in range(len(self.data.linear.rfa.poles)):
                    D[i_mat+3, :, :] = self.data.linear.rfa.matrices_w[i_mat+1][:, [3*i for i in leading_edge_index]]

                # Collocation point coordinates and dihedral for leading edge panels
                col, dih = self.calculate_collocation_dihedral()
                col_le = col[leading_edge_index, :]
                dih_le = dih[leading_edge_index]

                inp.systems.sett.s1.aero.D = jnp.array(D)
                inp.systems.sett.s1.aero.gust.panels_dihedral = jnp.array(dih_le)
                inp.systems.sett.s1.aero.gust.collocation_points = jnp.array(col_le)

                inp.systems.sett.s1.aero.gust_profile = "mc"
                inp.systems.sett.s1.aero.gust.intensity = self.settings['gust_intensity']
                inp.systems.sett.s1.aero.gust.length = self.settings['gust_length']
                inp.systems.sett.s1.aero.gust.step = self.settings['gust_length']/self.settings['gust_num_x']
                inp.systems.sett.s1.aero.gust.shift = self.settings['gust_offset']

        # Statespace aero inputs
        elif self.settings['aero_approx'] == 'statespace' and self.settings['aero_on']:
            # Remove structural states
            states_keep = []
            for i_s in range(self.data.linear.ss.state_variables.num_variables):
                if self.data.linear.ss.state_variables.vector_variables[i_s].name not in ['q', 'q_dot']:
                    states_keep += list(self.data.linear.ss.state_variables.vector_variables[i_s].cols_loc)
            
            # Remove non-forcing outputs
            outputs_keep = []
            for i_s in range(self.data.linear.ss.output_variables.num_variables):
                if self.data.linear.ss.output_variables.vector_variables[i_s].name == 'Q':
                    outputs_keep += list(self.data.linear.ss.output_variables.vector_variables[i_s].rows_loc)

            # Truncate states and outputs
            ss_d_trunc = pyyeti.ssmodel.SSModel(self.data.linear.ss.A[np.ix_(states_keep, states_keep)], \
                                    self.data.linear.ss.B[states_keep, :], \
                                    self.data.linear.ss.C[np.ix_(outputs_keep, states_keep)], \
                                    self.data.linear.ss.D[outputs_keep, :], \
                                    self.data.linear.ss.dt)
            
            # Convert to continuous time state space model
            ss_c = ss_d_trunc.d2c(self.settings['d2c_method'])

            # Split into three state space systems for each input
            # The A and C matrices are constant between the three
            for i_s in range(self.data.linear.ss.input_variables.num_variables):
                var_name = self.data.linear.ss.input_variables.vector_variables[i_s].name
                param_index = self.data.linear.ss.input_variables.vector_variables[i_s].cols_loc
                match var_name:
                    case 'q':
                        B0 = ss_c.B[:, param_index]
                        D0 = ss_c.D[:, param_index]
                    case 'q_dot':
                        B1 = ss_c.B[:, param_index]
                        D1 = ss_c.D[:, param_index]
                    case 'u_gust':
                        Bw = ss_c.B[:, param_index]
                        Dw = ss_c.D[:, param_index]
            
            inp.systems.sett.s1.aero.approx = 'statespace'
            inp.systems.sett.s1.aero.use_reduced_time = self.settings['reduced_time_ss']
            inp.systems.sett.s1.aero.ss_A = jnp.array(ss_c.A, dtype=float)
            inp.systems.sett.s1.aero.ss_B0 = jnp.array(B0, dtype=float)
            inp.systems.sett.s1.aero.ss_B1 = jnp.array(B1, dtype=float)
            inp.systems.sett.s1.aero.ss_C = jnp.array(ss_c.C, dtype=float)
            inp.systems.sett.s1.aero.ss_D0 = jnp.array(D0, dtype=float)
            inp.systems.sett.s1.aero.ss_D1 = jnp.array(D1, dtype=float)

            # Aero due to disturbances
            if self.settings['gust_on']:

                # Find index of leading edge elements in state space gust vector
                leading_edge_index = []
                prev_end = 0
                for surf_dim in self.data.aero.timestep_info[self.settings['use_custom_timestep']].dimensions:
                    leading_edge_index.extend((np.arange(surf_dim[1]+1)*(surf_dim[0]+1))+prev_end)
                    prev_end += (surf_dim[0]+1)*(surf_dim[1]+1)

                # Create gust matrices for leading edge panels only
                n_w = len(leading_edge_index)

                Bw_le = Bw[:, [3*i for i in leading_edge_index]]
                Dw_le = Dw[:, [3*i for i in leading_edge_index]]

                # Collocation point coordinates and dihedral for leading edge panels
                col, dih = self.calculate_collocation_dihedral()
                col_le = col[leading_edge_index, :]
                dih_le = dih[leading_edge_index]

                inp.systems.sett.s1.aero.ss_Bw = jnp.array(Bw_le)
                inp.systems.sett.s1.aero.ss_Dw = jnp.array(Dw_le)
                inp.systems.sett.s1.aero.gust.panels_dihedral = jnp.array(dih_le)
                inp.systems.sett.s1.aero.gust.collocation_points = jnp.array(col_le)

                inp.systems.sett.s1.aero.gust_profile = "mc"
                inp.systems.sett.s1.aero.gust.intensity = self.settings['gust_intensity']
                inp.systems.sett.s1.aero.gust.length = self.settings['gust_length']
                inp.systems.sett.s1.aero.gust.step = self.settings['gust_length']/self.settings['gust_num_x']
                inp.systems.sett.s1.aero.gust.shift = self.settings['gust_offset']

        # Determine number of time steps
        if self.settings['tn'] > 0:     # Set number of steps from input
            tn = self.settings['tn']
        else:                           # Set number of steps from eigenvalues
            (evals, _) = np.linalg.eig(K @ np.linalg.inv(M))
            evals.sort()

            if self.settings['aero_approx'] == 'roger':
                tn = int(2*np.sqrt(evals[self.settings['num_modes']-1])*self.settings['t_factor']*self.settings['t1'])
            elif self.settings['aero_approx'] == 'statespace':
                tn_struct = int(2*np.sqrt(evals[self.settings['num_modes']-1]))
                tn_aero = int(np.max(np.linalg.eig(ss_c.A)[0].imag)*(2*u_inf)/self.settings['c_ref'])
                cout.cout_wrap(f"Required structure time steps per second: {tn_struct}", 1)
                cout.cout_wrap(f"Required aero time steps per second: {tn_aero}", 1)
                tn = int(max((tn_struct, tn_aero))*self.settings['t_factor']*self.settings['t1'])
            else:
                raise AttributeError

        inp.systems.sett.s1.tn = tn
        cout.cout_wrap(f"Number of time steps: {tn}", 0)

        return inp