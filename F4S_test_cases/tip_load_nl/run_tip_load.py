import shutil
import os
import warnings
import numpy as np
from scipy.signal import StateSpace, dlsim
from matplotlib import pyplot as plt

import sharpy.sharpy_main
from wing_generator import swept_tip_goland


# Clear interpreter console
os.system('cls' if os.name == 'nt' else 'clear')

# Disable warnings for more readable output
warnings.filterwarnings("ignore")

# Remove old case files and outputs
try:
    shutil.rmtree('' + str(os.path.dirname(os.path.realpath(__file__))) + '/output/')
    shutil.rmtree('' + str(os.path.dirname(os.path.realpath(__file__))) + '/cases/')
except:
    pass

case_output = dict()
case_data = dict()

flow = [
    'BeamLoader',
    'AerogridLoader',
    'Modal',
    # 'StaticCoupled',
    'DynamicCoupled',
    # 'StaticUvlm',
    # 'LinearAssembler',
    # 'AsymptoticStability',
    # 'LinearRFA',
    'Intrinsic',
    'IntrinsicPlot',
    'LinearAssembler'
]

# Case parameters
c_ref = 1.8288
ar = 7
num_modes = 15
u_inf = 40. # 75.0
physical_time = 0.25

gust_on = False
aero_on = False
gravity_on = True
# tip_f = [0., 0., 5e4]
# tip_f = [0., 0., 5e2]
tip_f = [0., 0., 0.]

M = 12  # 8 #12  #6
N = 30  # 40 #20
alpha = np.deg2rad(60.)
beta = np.deg2rad(45.)
ang_h = np.deg2rad(0.)  # 15

linear = False

b_ref = c_ref * ar

case_name = 'beam_test'
# Generate wing and run SHARP
wing = swept_tip_goland(case_name, flow,
                        disc_mode=5,
                        c_ref=c_ref,
                        ang_h=ang_h,
                        pos_frac_h=0.7,
                        u_inf=u_inf,
                        n_surf=1,
                        b_ref=c_ref * ar,
                        alpha=alpha,
                        beta=beta,
                        physical_time=physical_time,
                        M=M,
                        N=N,
                        beam_dir='y',
                        tip_f=tip_f,
                        Mstar_fact=2.5,
                        wake_cfl1=False,
                        gravity_on=gravity_on,
                        write_screen='on',
                        lin_tstep=0,
                        lin_n_modes=num_modes,
                        lin_system='LinearAeroelastic' if aero_on else 'LinearBeam',
                        n_modes_modal=num_modes,
                        itsc_aero_approx='statespace',
                        itsc_dt_fact=0.1,
                        sigma_1=0.5,
                        main_ea=0.5,
                        main_cg=0.5,
                        # sigma_2 = 0.03,
                        # itsc_dt_fact = 1.0,
                        itsc_n_tstep=0,
                        # itsc_nonlinear=-1 if linear else 1,  #1 is nonlinear, -1 is linear
                        # itsc_nonlinear=0,
                        # itsc_nonlinear=-1,
                        itsc_nonlinear=1,
                        itsc_q0_treatment=2,
                        itsc_d2c_method='tustin',
                        aero_on=aero_on,
                        gust_on=gust_on,
                        rom_method='Krylov',
                        krylov_r=6,
                        # lin_coords = 'nodes',
                        # gust_intensity = 0.1,
                        gust_intensity=0.05 if gust_on else 0.0,
                        gust_length=0.5 * u_inf,
                        gust_offset=0.25 * u_inf,
                        mode_sign_convention=True)

case_data = sharpy.sharpy_main.main(['', wing.route + wing.case_name + '.sharpy'])

z_dc = np.array([ts.pos[-1, 2] for ts in case_data.structure.timestep_info])
z_itsc = case_data.intrinsic.r_a[:, -1, 2]

# linear
a = case_data.linear.ss.A
b = case_data.linear.ss.B
c = case_data.linear.ss.C
d = case_data.linear.ss.D
dt_lin = case_data.linear.ss.dt
evects = case_data.structure.timestep_info[0].modal['eigenvectors']     # [num_nodes x num_modes]
t_s = np.arange(0., physical_time, dt_lin)

i_in_Q = case_data.linear.ss.input_variables[0].cols_loc
i_out_q = case_data.linear.ss.output_variables[0].rows_loc

f_tip = np.zeros(evects.shape[0])
f_tip[-6:-3] = tip_f

eta_f_tip = evects.T @ f_tip

u_in = np.zeros((len(t_s), b.shape[1]))
u_in[:, i_in_Q] = eta_f_tip

sys = StateSpace(a, b, c, d, dt=dt_lin)
_, q_out, x_out = dlsim(sys, u_in, t_s)

z_lin = (q_out[:, i_out_q] @ evects.T)[:, -4]

fig, ax = plt.subplots()
ax.plot(np.linspace(0., physical_time, case_data.ts + 1), z_dc / b_ref, label='Dynamic Coupled')
ax.plot(case_data.intrinsic.t, z_itsc / b_ref, label="Intrinsic")
ax.plot(t_s, z_lin / b_ref, label='Linear')
ax.legend()
plt.xlabel("Time (s)")
plt.ylabel("Tip Vertical Displacement (m)")
plt.show()
pass
