import os
import numpy as np
import configobj

import sharpy.sharpy_main
from sharpy.utils.algebra import euler2quat
from pazy_wing_model import PazyWing

output_folder = './output/'
case_route = './cases/'

if not os.path.exists(case_route):
    os.makedirs(case_route)

num_modes = 20
alpha = np.deg2rad(1e-5)
quat = euler2quat((0., alpha, 0.))
u_inf = 20.
u_inf_dir = [1., 0., 0.]
rho = 1.225
c_ref = 0.1
m = 8
m_star_fact = 2
dt = c_ref / (u_inf * m)
physical_time = 1.
n_tstep = int(physical_time // dt)
itsc_dt_fact = 1.
gust_length = 1.
gust_intensity = 0.5
gust_offset = 0.
gravity_on = True
num_surfaces = 1

# for sweep_angle in np.deg2rad((0., 20., 30.)):
for sweep_angle in np.deg2rad(np.array((0., ))):
    case_name = f'swept_pazy_wing_{np.rad2deg(sweep_angle):.0f}'
    print(case_name)

    model_settings = {
        'skin_on': 'off',
        'discretisation_method': 'michigan',
        # 'discretisation_method': 'even',
        # 'discretisation_method': 'fine_root_tip',
        'surface_m': m,
        'num_elem': 2,
        'num_surfaces': num_surfaces,
        'sweep_angle': sweep_angle
    }

    pazy = PazyWing(case_name, case_route, in_settings=model_settings)
    pazy.generate_structure()
    if num_surfaces == 2:
        pazy.structure.mirror_wing()
    pazy.generate_aero()
    pazy.save_files()

    settings = dict()

    config = configobj.ConfigObj()
    config.filename = f'./{case_name}.sharpy'

    settings['SHARPy'] = {
        'flow': [
            'BeamLoader',
            'AerogridLoader',
            'Modal',
            # 'DynamicCoupled'
            'StaticUvlm',
            # 'BeamPlot',
            # 'AerogridPlot',
            'LinearAssembler',
            'Intrinsic',
            'IntrinsicPlot'
        ],
        'case': case_name,
        'route': case_route,
        'write_screen': True,
        'write_log': 'on',
        'log_folder': output_folder + '/' + case_name + '/',
        'log_file': case_name + '.log'}

    settings['BeamLoader'] = {
        'unsteady': True,
        'orientation': quat}

    settings['AerogridLoader'] = {
        'unsteady': True,
        'aligned_grid': True,
        'mstar': m * m_star_fact,
        'freestream_dir': u_inf_dir,
        'wake_shape_generator': 'StraightWake',
        'wake_shape_generator_input': {'u_inf': u_inf,
                                       'u_inf_direction': u_inf_dir,
                                       'dt': dt}}

    settings['Modal'] = {
        'NumLambda': num_modes,
        'rigid_body_modes': False,
        'print_matrices': True,
        'continuous_eigenvalues': False,
        'dt': 0,
        'plot_eigenvalues': False,
        'write_modes_vtk': True,
        'use_undamped_modes': True,
        # 'mode_sign_convention': False
    }

    settings['StaticUvlm'] = {
        'print_info': True,
        'horseshoe': False,
        'num_cores': 8,
        'velocity_field_generator': 'SteadyVelocityField',
        'velocity_field_input': {'u_inf': u_inf,
                                 'u_inf_direction': u_inf_dir},
        'rho': rho,
        'cfl1': True}

    settings['BeamPlot'] = {}

    settings['AerogridPlot'] = {
        'include_rbm': False,
        'include_applied_forces': True,
        'minus_m_star': 0}

    settings['DynamicCoupled'] = {'structural_solver': 'NonLinearDynamicPrescribedStep',
                                  'structural_solver_settings': {'print_info': False,
                                                                 'gravity_on': gravity_on,
                                                                 'gravity': 9.81,
                                                                 'num_steps': n_tstep,
                                                                 'dt': dt,
                                                                 },

                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': {'print_info': False,
                                                           'num_cores': 8,
                                                           'rho': rho,
                                                           'convection_scheme': 2,
                                                           'gamma_dot_filtering': 6,
                                                           'n_time_steps': n_tstep,
                                                           'dt': dt,
                                                           'cfl1': True,
                                                           'velocity_field_generator': 'GustVelocityField',
                                                           'velocity_field_input':
                                                               {'u_inf': u_inf,
                                                                'u_inf_direction': u_inf_dir,
                                                                'gust_shape': '1-cos',
                                                                'gust_parameters': {
                                                                    'gust_length': gust_length,
                                                                    'gust_intensity': gust_intensity * u_inf},
                                                                'offset': gust_offset,
                                                                'relative_motion': 'on'}},
                                  'n_time_steps': n_tstep,
                                  'dt': dt,
                                  'include_unsteady_force_contribution': True,
                                  'postprocessors': ['BeamPlot', 'AerogridPlot'],
                                  'postprocessors_settings': {'BeamPlot': {'include_rbm': True,
                                                                           'include_applied_forces': True},
                                                              'AerogridPlot': {
                                                                  'include_rbm': True,
                                                                  'include_applied_forces': True,
                                                                  'minus_m_star': 0},
                                                              }}

    settings['LinearAssembler'] = {
        'linear_system': 'LinearAeroelastic',
        'linearisation_tstep': 0,
        'modal_tstep': 0,
        'inout_coordinates': 'modes',
        'linear_system_settings': {
            'beam_settings': {'modal_projection': True,
                              'inout_coords': 'modes',
                              'discrete_time': True,
                              'newmark_damp': 5e-5,
                              'discr_method': 'newmark',
                              'dt': dt,
                              'proj_modes': 'undamped',
                              'use_euler': False,
                              'num_modes': num_modes,
                              'print_info': True,
                              'gravity': True,
                              'remove_sym_modes': False,
                              'remove_dofs': []},
            'aero_settings': {'dt': dt,
                              'integr_order': 2,
                              'density': rho,
                              'remove_predictor': True,
                              'use_sparse': True,
                              }}}

    settings['Intrinsic'] = {'num_modes': num_modes,
                             'orientation': quat,
                             'aero_approx': 'statespace',
                             't1': physical_time,
                             'dt': dt * itsc_dt_fact,
                             'c_ref': c_ref,
                             'rho': rho,
                             'u_inf': u_inf,
                             'gravity_on': True,
                             'aero_on': True,
                             'gust_on': True,
                             'gust_intensity': gust_intensity * u_inf,
                             'gust_offset': gust_offset,
                             'gust_length': gust_length,
                             'gust_num_x': 30}

    settings['IntrinsicPlot'] = {}

    for k, v in settings.items():
        config[k] = v
    config.write()

    case_data = sharpy.sharpy_main.main(['', case_name + '.sharpy'])

    pass
