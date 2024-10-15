import shutil
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

import sharpy.sharpy_main

from helper_functions.case_data_extract import case_data_extract
from wing_generator import swept_tip_goland

### Disable warnings for more readable output
warnings.filterwarnings("ignore")    

### Remove old case files and outputs
# try:
#         shutil.rmtree('' + str(os.path.dirname(os.path.realpath(__file__))) + '/output/')
#         shutil.rmtree('' + str(os.path.dirname(os.path.realpath(__file__))) + '/cases/')
# except: ()

case_output = dict()

flow =  ['BeamLoader', 
        'AerogridLoader',
        'StaticCoupled',
        'Modal',
        'AerogridPlot',
        'BeamPlot',
        'DynamicCoupled',
        'AeroForcesCalculator',
        # 'LinearAssembler',
        # 'AsymptoticStability',
        ]

sweep = np.deg2rad(60)
u_inf = 50       
c_ref = 1.8288
ar = 10   

### Loop this code, varying parameters each run
for i_case in range(3):
        case_name = ['sidesweep', 'aligned', 'misaligned'][i_case]

        wing = swept_tip_goland(case_name, flow,
                                c_ref = [c_ref, c_ref/np.cos(sweep), c_ref][i_case],
                                ang_panel = [0, 0, -sweep][i_case],
                                Mstar_fact = [10, 10, 10.0/np.cos(sweep)][i_case],
                                sweep = [0, sweep, sweep][i_case],
                                beta = [sweep, 0, 0][i_case],
                                u_inf = u_inf,
                                n_surf = 1,
                                b_ref = c_ref*ar,
                                gust_intensity = 0.5,
                                gust_length = 0.005 * u_inf,
                                gust_offset = 0.0025 * u_inf,
                                physical_time = 1.5,
                                sigma = 1e6,
                                sym = False)

        case_data = sharpy.sharpy_main.main(['', wing.route + wing.case_name + '.sharpy'])
        case_output.update({wing.case_name: case_data_extract(wing, case_data)})
### End of loop

spio.savemat('case_output.mat', case_output)