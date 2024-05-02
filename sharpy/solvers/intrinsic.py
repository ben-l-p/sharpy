# Python package imports
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
import warnings
import os

# General SHARPy imports
import sharpy.solvers._basestructural as basestructuralsolver
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.structure.utils.xbeamlib as xbeamlib
from sharpy.utils.cout_utils import cout_wrap

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
    settings_default['num_modes'] = 20
    settings_description['num_modes'] = 'Number of modes to retain'

    settings_types['delta_curved'] = 'float'
    settings_default['delta_curved'] = 1e-2
    settings_description['delta_curved'] = 'Threshold for linear rotations'

    settings_types['plot_eigenvalues'] = 'bool'
    settings_default['plot_eigenvalues'] = False
    settings_description['plot_eigenvalues'] = 'Plot to screen root locus diagram'

    settings_types['use_custom_timestep'] = 'int'
    settings_default['use_custom_timestep'] = -1
    settings_description['use_custom_timestep'] = 'If > -1, it will use that time step geometry for calculating the modes'

    settings_types['component_names'] = 'list(str)'
    settings_default['component_names'] = []
    settings_description['component_names'] = 'Name components of the structure. Will use lettering [A, B, ...] by default'

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
    settings_default['solution'] = 'static'
    settings_description['solution'] = 'Solution type to be used in FEM4INAS'

    settings_types['solver_library'] = 'str'
    settings_default['solver_library'] = 'diffrax'
    settings_description['solver_library'] = 'Solver library to be used in FEM4INAS'

    settings_types['solver_function'] = 'str'
    settings_default['solver_function'] = 'newton_raphson'
    settings_description['solver_function'] = 'Solver function to be used in FEM4INAS'

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

        # Select timestep for initialisation
        self.data.ts = -1
        if int(self.settings['use_custom_timestep']) > -1:
            self.data.ts = self.settings['use_custom_timestep']

    def run(self, **kwargs):
        FEM_route = self.generate_FEM_files()
        input = self.generate_settings_file(FEM_route)
        config = Config(input)
        cout_wrap("Running FEM4INAS\n")
        sol = fem4inas_main.main(input_obj=config)
        self.intrinsic_output_convert(sol)

    # Returns the global mass and stiffness matrices
    def get_M_C_K(self, n_dof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        M_full = np.zeros([n_dof, n_dof],
                            dtype=ct.c_double, order='F')
        C_full = np.zeros([n_dof, n_dof],
                            dtype=ct.c_double, order='F')
        K_full = np.zeros([n_dof, n_dof],
                            dtype=ct.c_double, order='F')

        xbeamlib.cbeam3_solv_modal(self.data.structure, self.settings, self.data.ts,
                                        M_full, K_full, C_full)

        return np.array(M_full), np.array(K_full), np.array(C_full)

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
        pos = self.data.structure.timestep_info[self.data.ts].pos
        connectivity = self.data.structure.connectivities
        beam_number = self.data.structure.beam_number
        return pos, connectivity, beam_number 

    def generate_FEM_files(self) -> str:
        cout_wrap("Generating input FEM data")
        # Save structure matrices
        n_dof = self.data.structure.num_dof.value
        M_full, C_full, K_full = self.get_M_C_K(n_dof)
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
        
        cout_wrap(f'\tFEM files generated in {FEM_route}', 1)
        return FEM_route

    def generate_settings_file(self, FEM_route) -> Inputs:
        ints_output_folder = self.data.output_folder + 'intrinsic/'
        os.mkdir(ints_output_folder)

        inp = Inputs()
        inp.engine = self.settings['engine']
        inp.fem.connectivity = dict(A=None)             #TODO: replace with connectivity
        inp.fem.folder = FEM_route
        inp.fem.num_modes = self.settings['num_modes']
        inp.driver.typeof = self.settings['driver']
        inp.driver.sol_path = ints_output_folder
        inp.simulation.typeof = self.settings['sim_type']
        inp.systems.sett.s1.solution = self.settings['solution']
        inp.systems.sett.s1.solver_library = self.settings['solver_library']
        inp.systems.sett.s1.solver_function = self.settings['solver_function']
        inp.systems.sett.s1.solver_settings = dict(rtol=self.settings['rtol'],
                                                atol=self.settings['atol'],
                                                max_steps=self.settings['max_steps'],
                                                norm=self.settings['norm'],
                                                kappa=self.settings['kappa'])
        
        inp.systems.sett.s1.xloads.follower_forces = True
        inp.systems.sett.s1.xloads.follower_points = [[25, 1]]
        inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6, 7]
        inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                            -3.7e3,
                                                            -12.1e3,
                                                            -17.5e3,
                                                            -39.3e3,
                                                            -61.0e3,
                                                            -94.5e3,
                                                            -120e3]
                                                            ]
        inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6, 7]

        return inp

    def intrinsic_output_convert(self, sol):
        pass
