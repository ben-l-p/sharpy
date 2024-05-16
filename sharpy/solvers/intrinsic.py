# Python package imports
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
import warnings
import os
import scipy.io as spio
import jax.numpy as jnp

# General SHARPy imports
import sharpy.solvers._basestructural as basestructuralsolver
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.structure.utils.xbeamlib as xbeamlib
import sharpy.utils.cout_utils as cout

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

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Write status to screen'

    settings_types['num_modes'] = 'int'
    settings_default['num_modes'] = None
    settings_description['num_modes'] = 'Number of modes to retain'

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

    # Settings to be passed to FEM4INAS
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
    settings_default['tn'] = None
    settings_description['tn'] = 'Number of simulation steps'

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

    settings_types['aero_on'] = 'bool'
    settings_default['aero_on'] = True
    settings_description['aero_on'] = 'Enable aerodynamics'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.data = None
        self.settings = None

    def initialise(self, data, custom_settings=None, restart=False):
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

    def run(self, **kwargs):
        intrinsic_out = self.Intrinsic_Obj()
        FEM_route = self.generate_FEM_files()

        # Stability analysis
        if self.settings['stability_analysis']:
            cout.cout_wrap("Running stability analysis", 0)
            stab_out = []
            vels = np.linspace(self.settings['stability_v_min'],
                               self.settings['stability_v_max'], 
                               self.settings['stability_v_num'])
            for vel in vels:
                input = self.generate_settings_file(FEM_route, float(vel))
                config = Config(input)
                cout.cout_wrap(f"\nVelocity: {vel:.2f} m/s", 1)
                cout.cout_wrap("Running Intrinsic Solver", 0)
                sol = fem4inas_main.main(input_obj=config)
                cout.cout_wrap("Intrinsic Solution Complete", 0)

                if jnp.any(jnp.isnan(sol.dynamicsystem_s1.q)):
                    stab_out.append({'u_inf': vel, 'is_stable': False})
                    cout.cout_wrap("    Stable: False", 1)
                else:
                    stab_out.append({'u_inf': vel, 'is_stable': True})
                    cout.cout_wrap("    Stable: True", 1)

        # Case to save
        input = self.generate_settings_file(FEM_route, self.settings['u_inf'])
        config = Config(input)
        cout.cout_wrap("\nRunning Intrinsic Solver", 0)
        sol = fem4inas_main.main(input_obj=config)
        cout.cout_wrap("Intrinsic Solution Complete", 0)

        if jnp.any(jnp.isnan(sol.dynamicsystem_s1.q)):
            cout.cout_wrap("    Warning - model is unstable", 1)

        intrinsic_out.update_sol(sol)
        self.data.intrinsic = intrinsic_out
        return self.data

    # Returns the global mass and stiffness matrices
    def get_M_C_K(self, n_dof: int, tstep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        M_full = np.zeros([n_dof, n_dof],
                            dtype=ct.c_double, order='F')
        C_full = np.zeros([n_dof, n_dof],
                            dtype=ct.c_double, order='F')
        K_full = np.zeros([n_dof, n_dof],
                            dtype=ct.c_double, order='F')

        xbeamlib.cbeam3_solv_modal(self.data.structure, self.settings, tstep,
                                        M_full, C_full, K_full)

        return np.array(M_full), np.array(C_full), np.array(K_full)

    # Returns the eigenvalues and eigenvectors of the system
    def get_eigs0(self, n_dof, M_full, C_full, K_full) -> tuple[np.ndarray, np.ndarray]:
        # Check if the system has damping
        if np.max(np.abs(C_full)) > np.finfo(float).eps:
            warnings.warn('Projecting a system with damping on undamped modal shapes')

        # Solve eigen problem (no damping)
        [evals,evects] = np.linalg.eig(np.linalg.solve(M_full, K_full))

        # Sort eigenvalues/vectors in frequency order
        NumLambda = min(n_dof, int(self.settings['num_modes']))
        order = np.argsort(np.sqrt(evals/1j))[:NumLambda]
        evals = evals[order]
        evects = evects[:,order]

        # Plot eigenvalues using matplotlib if specified in settings
        if self.settings['plot_eigenvalues']:
            plt.figure()
            plt.scatter(np.zeros_like(evals), evals)
            plt.show()

        return evals, evects

    # Returns the grid nodes of the system
    def get_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = self.data.structure.timestep_info[self.settings['use_custom_timestep']].pos
        connectivity = self.data.structure.connectivities
        beam_number = self.data.structure.beam_number
        return pos, connectivity, beam_number 

    def generate_FEM_files(self) -> str:
        cout.cout_wrap("Generating input FEM data")
        # Save structure matrices
        n_dof = self.data.structure.num_dof.value
        M_full, C_full, K_full = self.get_M_C_K(n_dof, self.settings['use_custom_timestep'])
        evals, evects = self.get_eigs0(n_dof, M_full, C_full, K_full)

        FEM_route = self.data.case_route + 'intrinsic_FEM/'
        os.mkdir(FEM_route)
        np.save(FEM_route + 'Ma', M_full)
        np.save(FEM_route + 'Ka', K_full)
        np.save(FEM_route + 'eigenvecs', evects)
        np.save(FEM_route + 'eigenvals', evals)

        # Create structural grid
        pos, connectivity, beam_number = self.get_grid()
        n_elem = pos.shape[0]

        n_beams = np.max(beam_number) + 1
        if len(self.settings['component_names']) != 0:      # TODO: fix the issue with settings passed as strings
            assert n_beams == len(self.settings['component_names']), "Number of component names does not match number of components"
            element_names = self.settings['component_names']
        else:
            element_names = [chr(65+i) for i in range(n_beams)]
            
        node_names = [element_names[0]] + [element_names[beam_number[i//2]] for i in range(n_elem-1)]       #TODO: make this use connectivity

        with open(FEM_route + 'structuralGrid', 'w') as f:
            f.write("# TITLE = \"Structural Grid\"\n")
            f.write("# VARIABLES = \"x\" \"y\" \"z\" \"FEM id\" \"Component\"")
            for i_node in range(n_elem):
                f.write(f"\n{pos[i_node, 0]} {pos[i_node, 1]} {pos[i_node, 2]} {i_node - 1} {node_names[i_node]}")
        f.close()
        
        cout.cout_wrap(f'\tFEM files generated in {FEM_route}', 1)
        return FEM_route
    
    # def q0_init(self, evects) -> list:
    #     tstep_q0 = self.settings['dynamic_tstep_init']
    #     n_node = self.data.structure.num_node
    #     x0 = self.data.structure.timestep_info[tstep_q0].q[:(n_node-1)*6]
    #     q0 = np.linalg.lstsq(evects, x0)[0]
    #     return list(q0)

    def generate_settings_file(self, FEM_route, u_inf) -> Inputs:
        ints_output_folder = self.data.output_folder + 'intrinsic/'

        try:
            os.mkdir(ints_output_folder)
        except: 
            pass

        inp = Inputs()
        inp.engine = self.settings['engine']
        inp.fem.connectivity = dict(A=['B'], B = None)             #TODO: replace with connectivity
        inp.fem.folder = FEM_route
        inp.fem.num_modes = self.settings['num_modes']
        inp.driver.typeof = self.settings['driver']
        inp.driver.sol_path = ints_output_folder
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
        
        inp.systems.sett.s1.t1 = self.settings['t1']
        inp.systems.sett.s1.tn = self.settings['tn']
        inp.systems.sett.s1.aero.rho_inf = self.settings['rho']
        inp.systems.sett.s1.aero.u_inf = u_inf
        inp.systems.sett.s1.aero.c_ref = self.settings['c_ref']
        inp.systems.sett.s1.xloads.modalaero_forces = self.settings['aero_on']
        inp.systems.sett.s1.xloads.gravity_forces = self.settings['gravity_on']

        # Aero due to structure
        A = np.zeros([3 + len(self.data.linear.rfa.poles), self.settings['num_modes'], self.settings['num_modes']], dtype=float)
        A[0, :, :] = self.data.linear.rfa.matrices_q[0]
        A[1, :, :] = self.data.linear.rfa.matrices_q[1]
        for i_mat in range(len(self.data.linear.rfa.poles)):
            A[i_mat+3, :, :] = self.data.linear.rfa.matrices_q[i_mat+2]

        try:
            os.mkdir(self.data.case_route + 'roger')
        except:
            pass

        np.save(self.data.case_route + 'roger/poles.npy', -np.array(self.data.linear.rfa.poles))
        np.save(self.data.case_route + 'roger/A.npy', A)
        inp.systems.sett.s1.aero.poles = self.data.case_route + 'roger/poles.npy'
        inp.systems.sett.s1.aero.A = self.data.case_route + 'roger/A.npy'

        # Aero due to disturbances
        if self.data.linear.rfa.matrices_w is not None:
            n_w = self.data.linear.rfa.matrices_w[0].shape[1]
            D = np.zeros([3 + len(self.data.linear.rfa.poles), self.settings['num_modes'], n_w], dtype=float)
            D[0, :, :] = self.data.linear.rfa.matrices_w[0]
            for i_mat in range(len(self.data.linear.rfa.poles)):
                D[i_mat+3, :, :] = self.data.linear.rfa.matrices_w[i_mat+1]

            np.save(self.data.case_route + 'roger/D.npy', D)
            inp.systems.sett.s1.aero.D = self.data.case_route + 'roger/D.npy'

        return inp