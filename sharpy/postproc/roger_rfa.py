import numpy as np
import pyyeti.ssmodel
import scipy.optimize
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import pyyeti
import scipy
import itertools
import matplotlib.pyplot as plt
import sharpy.utils.cout_utils as cout

@solver
class Roger_RFA(BaseSolver):
    """
    Plots the flow field in Paraview and computes the velocity at a set of points in a grid.
    """
    solver_id = 'RogerRFA'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()
    settings_options = dict()
    
    settings_types['rfa_type'] = 'str'
    settings_default['rfa_type'] = 'roger'
    settings_description ['rfa_type'] = 'RFA to be used'
    settings_options['rfa_type'] = ['roger', 'eversman']

    settings_types['num_poles'] = 'int'
    settings_default['num_poles'] = 3
    settings_description ['num_poles'] = 'Number of poles to fit for approximation.'

    settings_types['imag_weight'] = 'float'
    settings_default['imag_weight'] = 1.0
    settings_description ['imag_weight'] = 'Add or reduce effect of imaginary component relative to real component in fit.'

    settings_types['k_min'] = 'float'
    settings_default['k_min'] = 1e-4
    settings_description ['k_min'] = 'Minimum reduced frequency for sampling data'

    settings_types['k_max'] = 'float'
    settings_default['k_max'] = 1e1
    settings_description ['k_max'] = 'Maximum reduced frequency for sampling data'

    settings_types['k_num'] = 'int'
    settings_default['k_num'] = 50
    settings_description ['k_num'] = 'Number of reduced frequency points to sample data'

    settings_types['k_spacing'] = 'str'
    settings_default['k_spacing'] = 'log'
    settings_description ['k_spacing'] = 'Spacing of reduced frequency points for sampling data'
    settings_options['k_spacing'] = ['lin', 'log']

    settings_types['p_min'] = 'float'
    settings_default['p_min'] = 1e-2
    settings_description ['p_min'] = 'Minimum absolute pole value for discrete optimisation'

    settings_types['p_max'] = 'float'
    settings_default['p_max'] = 1e4
    settings_description ['p_max'] = 'Maximum absolute pole value for discrete optimisation'

    settings_types['p_num'] = 'int'
    settings_default['p_num'] = 20
    settings_description ['p_num'] = 'Number of pole values for discrete optimisation'

    settings_types['p_spacing'] = 'str'
    settings_default['p_spacing'] = 'log'
    settings_description ['p_spacing'] = 'Spacing of pole values for discrete optimisation'
    settings_options['p_spacing'] = ['lin', 'log']

    settings_types['p_input'] = 'list(float)'
    settings_default['p_input'] = []
    settings_description ['p_input'] = 'Manually input negative poles (these will not be optimised)'

    settings_types['fit_u_gust'] = 'bool'
    settings_default['fit_u_gust'] = True
    settings_description ['fit_u_gust'] = 'Calculate RFA matrices for disturbance terms'

    settings_types['d2c_method'] = 'str'
    settings_default['d2c_method'] = 'tustin'
    settings_description ['d2c_method'] = 'Method for converting state space from discrete to continuous time'
    settings_options['d2c_method'] = ['zoh', 'zoha', 'foh', 'tustin']

    settings_types['plot_rfa'] = 'bool'
    settings_default['plot_rfa'] = 'False'
    settings_description ['plot_rfa'] = 'Plot RFA'

    settings_types['plot_type'] = 'str'
    settings_default['plot_type'] = 'polar'
    settings_description ['plot_type'] = 'Plot RFA'
    settings_options['plot_type'] = ['bode', 'polar', 'real_imag']

    settings_types['num_q_plot'] = 'int'
    settings_default['num_q_plot'] = 2
    settings_description['num_q_plot'] = 'Plot RFA'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description, settings_options)

    def __init__(self):
        self.settings = None
        self.data = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings,
                            self.settings_types,
                            self.settings_default,
                            self.settings_options)


    def run(self, **kwargs):      
        n_k = self.settings['k_num']
        n_p = self.settings['num_poles']
        match self.settings['k_spacing']:
            case 'lin':
                k_vals = np.linspace(self.settings['k_min'], \
                                self.settings['k_max'], n_k)
            case 'log':
                k_vals = np.logspace(np.log10(self.settings['k_min']), \
                                np.log10(self.settings['k_max']), n_k)
            case _:
                raise NotImplementedError(f"Unrecognised frequency spacing setting: {self.settings['k_spacing']}")
        
        match self.settings['p_spacing']:
            case 'lin':
                poles_disc = -np.linspace(self.settings['p_min'], \
                                self.settings['p_max'], self.settings['p_num'])
            case 'log':
                poles_disc = -np.logspace(np.log10(self.settings['p_min']), \
                                    np.log10(self.settings['p_max']), self.settings['p_num'])
            case _:
                raise NotImplementedError(f"Unrecognised discrete pole spacing setting: {self.settings['p_spacing']}")
            
        if self.data.linear.ss is None:
            raise AttributeError("Linear state-space system not found")
        
        if self.data.linear.linear_system.settings['beam_settings']['inout_coords'] != 'modes':
            raise AttributeError("Linear state-space system must use 'inout_coords' = 'modes'")
        
        # Remove structural states
        states_keep = []
        for i_s in range(self.data.linear.ss.state_variables.num_variables):
            if self.data.linear.ss.state_variables.vector_variables[i_s].name not in ['q', 'q_dot']:
                states_keep += list(self.data.linear.ss.state_variables.vector_variables[i_s].cols_loc)
        n_s = len(states_keep)
        
        outputs_keep = []
        for i_s in range(self.data.linear.ss.output_variables.num_variables):
            if self.data.linear.ss.output_variables.vector_variables[i_s].name == 'Q':
                outputs_keep += list(self.data.linear.ss.output_variables.vector_variables[i_s].rows_loc)

        # Convert to continuous time state space model
        ss_d_trunc = pyyeti.ssmodel.SSModel(self.data.linear.ss.A[np.ix_(states_keep, states_keep)], \
                                self.data.linear.ss.B[states_keep, :], \
                                self.data.linear.ss.C[np.ix_(outputs_keep, states_keep)], \
                                self.data.linear.ss.D[outputs_keep, :], \
                                self.data.linear.ss.dt)
        
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

        # A0 matrix for RFA
        A0q = D0 + ss_c.C @ np.linalg.inv(-ss_c.A) @ B0

        # Sample transfer function at frequency values
        n_q = B0.shape[1]
        Qq_sample = np.zeros([n_q, n_q, n_k], dtype=complex)
        for i_k, k in enumerate(k_vals):
            inv_mat = np.linalg.inv(np.eye(n_s)*1j*k-ss_c.A)
            Qq_sample[:, :, i_k] = ss_c.C @ inv_mat @ B0 + D0 + 1j*k*(ss_c.C @ inv_mat @ B1 + D1)

        # Calculate b term for Ax = b
        Lq_ls = np.zeros([2*n_k*n_q, n_q], dtype=float)
        for i_k, k in enumerate(k_vals):
            Lq_ls[2*i_k*n_q:(2*i_k+1)*n_q, :] = np.real(Qq_sample[:, :, i_k] - A0q)
            Lq_ls[(2*i_k+1)*n_q:(2*i_k+2)*n_q, :] = (np.imag(Qq_sample[:, :, i_k]) - k*D1)*self.settings['imag_weight']

        def least_squares_q(poles):
            Aq_roger = []
            Aq_roger.append(A0q)
            Aq_roger.append(D1)
            Qq_roger = np.zeros_like(Qq_sample)

            R_ls = np.zeros([2*n_k*n_q, n_q*n_p], dtype=float)
            for i_p, pole in enumerate(poles):
                for i_k, k in enumerate(k_vals):
                    if self.settings['rfa_type'] == 'roger':
                        R_ls[2*i_k*n_q:(2*i_k+1)*n_q, i_p*n_q:(i_p+1)*n_q] = np.eye(n_q)*k**2/(pole**2 + k**2)
                        R_ls[(2*i_k+1)*n_q:(2*i_k+2)*n_q, i_p*n_q:(i_p+1)*n_q] = np.eye(n_q)*k*pole/(pole**2 + k**2)*self.settings['imag_weight']
                    elif self.settings['rfa_type'] == 'eversman':
                        R_ls[2*i_k*n_q:(2*i_k+1)*n_q, i_p*n_q:(i_p+1)*n_q] = np.eye(n_q)*pole/(pole**2 + k**2)
                        R_ls[(2*i_k+1)*n_q:(2*i_k+2)*n_q, i_p*n_q:(i_p+1)*n_q] = np.eye(n_q)*-k/(pole**2 + k**2)*self.settings['imag_weight']
                    else:
                        raise NotImplementedError
                    

            xq_ls = np.linalg.lstsq(R_ls, Lq_ls)[0]
            for i_p in range(n_p):
                Aq_roger.append(xq_ls[i_p*n_q:(i_p+1)*n_q, :])

            for i_k, k in enumerate(k_vals):
                Qq_roger[:, :, i_k] = Aq_roger[0] + 1j*k*Aq_roger[1]
                for i_p, pole in enumerate(poles):
                    if self.settings['rfa_type'] == 'roger':
                        Qq_roger[:, :, i_k] += Aq_roger[i_p+2]*1j*k/(1j*k+pole)
                    elif self.settings['rfa_type'] == 'eversman':
                        Qq_roger[:, :, i_k] += Aq_roger[i_p+2]/(1j*k+pole)
            
            # errq = np.sum(np.linalg.norm(np.abs((Qq_roger - Qq_sample)/Qq_sample), 'fro', (0, 1)))/n_k
            # errq = np.sum(np.abs((Qq_roger - Qq_sample)/Qq_sample))/Qq_sample.size
            errq = np.linalg.norm(np.linalg.norm(np.nan_to_num(np.abs((Qq_roger - Qq_sample)/Qq_sample)), 'fro', (0, 1)))
            
            return Qq_roger, Aq_roger, errq, R_ls

        def least_squares_q_err(poles):
            return least_squares_q(poles)[2]

        # Manual pole input
        if len(self.settings['p_input']) != 0:
            cout.cout_wrap('Fitting RFA using input poles', 0)
            [Qq_roger, Aq_roger, errq, R_ls] = least_squares_q(self.settings['p_input'])
            cout.cout_wrap(f"\tAveraged relative error: {errq}", 1)

        # Pole optimisation
        else:                                               
            # Discrete optimisation
            poles_all_comb = list(itertools.combinations(poles_disc, n_p))
            cout.cout_wrap('Discrete optimisation to fit RFA', 0)
            cout.cout_wrap(f"    Combinations: {len(poles_all_comb)}", 1)

            errq_min = 1e9
            poles_disc_min = 1e9*np.ones(n_p)

            for poles_comb in poles_all_comb:
                errq = least_squares_q_err(poles_comb)
                if errq < errq_min:
                    errq_min = errq
                    poles_disc_min = poles_comb

            cout.cout_wrap('Discrete optimisation complete', 0)
            cout.cout_wrap('Poles:', 1)
            for pole in poles_disc_min:
                cout.cout_wrap(f"    {pole:.4f}", 2)
            cout.cout_wrap(f"Averaged relative error: {errq_min:4f}", 1)

            # Gradient optimisation
            poles = list(scipy.optimize.fmin(least_squares_q_err, poles_disc_min))
            [Qq_roger, Aq_roger, errq, R_ls] = least_squares_q(poles)

            cout.cout_wrap('Gradient optimisation complete', 0)
            cout.cout_wrap('    Poles:', 1)
            for pole in poles:
                cout.cout_wrap(f"    {pole:.4f}", 2)
            cout.cout_wrap(f"Averaged relative error: {errq:4f}", 1)

        out_dict = {'poles': poles, 'matrices_q': Aq_roger, 'sampled_ss_q_tf': Qq_sample, 'sampled_rfa_q_tf': Qq_roger, 'k': k_vals, \
                     'err_q': errq, 'matrices_w': None, 'sampled_ss_w_tf': None, 'sampled_rfa_w_tf': None, 'err_w': None}

        # Fit gust matrices
        if self.settings['fit_u_gust']:
            cout.cout_wrap('Fitting gust matrices to poles', 0)
            
            n_w = Dw.shape[1]
            
            # Sample gust transfer function at frequency values
            Qw_sample = np.zeros([n_q, n_w, n_k], dtype=complex)
            for i_k, k in enumerate(k_vals):
                inv_mat = np.linalg.inv(np.eye(n_s)*1j*k-ss_c.A)
                Qw_sample[:, :, i_k] = ss_c.C @ inv_mat @ Bw + Dw

            # Least squares fit using optimised poles
            A0w = Dw + ss_c.C @ np.linalg.inv(-ss_c.A) @ Bw
            Lw_ls = np.zeros([2*n_k*n_q, n_w], dtype=float)
            for i_k, k in enumerate(k_vals):
                Lw_ls[2*i_k*n_q:(2*i_k+1)*n_q, :] = np.real(Qw_sample[:, :, i_k] - A0w)
                Lw_ls[(2*i_k+1)*n_q:(2*i_k+2)*n_q, :] = np.imag(Qw_sample[:, :, i_k])*self.settings['imag_weight']

            xw_ls = np.linalg.lstsq(R_ls, Lw_ls)[0]
            Aw_roger = []
            Aw_roger.append(A0w)
            for i_p in range(n_p):
                Aw_roger.append(xw_ls[i_p*n_q:(i_p+1)*n_q, :])

            Qw_roger = np.zeros_like(Qw_sample)
            for i_k, k in enumerate(k_vals):
                Qw_roger[:, :, i_k] = Aw_roger[0]
                for i_p, pole in enumerate(poles):
                    if self.settings['rfa_type'] == 'roger':
                        Qw_roger[:, :, i_k] += Aw_roger[i_p+1]*1j*k/(1j*k+pole)
                    elif self.settings['rfa_type'] == 'eversman':
                        Qw_roger[:, :, i_k] += Aw_roger[i_p+1]/(1j*k+pole)

            errw = np.linalg.norm(np.linalg.norm(np.nan_to_num(np.abs((Qw_roger - Qw_sample)/Qw_sample)), 'fro', (0, 1)))
            cout.cout_wrap(f"    Averaged relative error: {errw:4f}", 1)

            out_dict['matrices_w'] = Aw_roger
            out_dict['sampled_ss_w_tf'] = Qw_sample
            out_dict['sampled_rfa_w_tf'] = Qq_roger
            out_dict['err_w'] = errq

        # Plotting
        if self.settings['plot_rfa']:
            if self.settings['plot_type'] == 'bode':
                _, ax = plt.subplots(2*self.settings['num_q_plot'], self.settings['num_q_plot'], sharex='all')
                for i_q_out in range(self.settings['num_q_plot']):
                    for i_q_in in range(self.settings['num_q_plot']):
                        ax[2*i_q_out, i_q_in].plot(k_vals, np.abs(Qq_sample[i_q_out, i_q_in, :]))
                        ax[2*i_q_out, i_q_in].plot(k_vals, np.abs(Qq_roger[i_q_out, i_q_in, :]))
                        ax[2*i_q_out, i_q_in].set_xscale('log')
                        ax[2*i_q_out+1, i_q_in].plot(k_vals, np.angle(Qq_sample[i_q_out, i_q_in, :]))
                        ax[2*i_q_out+1, i_q_in].plot(k_vals, np.angle(Qq_roger[i_q_out, i_q_in, :]))
                        ax[2*i_q_out+1, i_q_in].set_xscale('log')

            if self.settings['plot_type'] == 'real_imag':
                _, ax = plt.subplots(2*self.settings['num_q_plot'], self.settings['num_q_plot'], sharex='all')
                for i_q_out in range(self.settings['num_q_plot']):
                    for i_q_in in range(self.settings['num_q_plot']):
                        ax[2*i_q_out, i_q_in].plot(k_vals, np.real(Qq_sample[i_q_out, i_q_in, :]))
                        ax[2*i_q_out, i_q_in].plot(k_vals, np.real(Qq_roger[i_q_out, i_q_in, :]))
                        ax[2*i_q_out, i_q_in].set_xscale('log')
                        ax[2*i_q_out+1, i_q_in].plot(k_vals, np.imag(Qq_sample[i_q_out, i_q_in, :]))
                        ax[2*i_q_out+1, i_q_in].plot(k_vals, np.imag(Qq_roger[i_q_out, i_q_in, :]))
                        ax[2*i_q_out+1, i_q_in].set_xscale('log')

            elif self.settings['plot_type'] == 'polar':
                _, ax = plt.subplots(self.settings['num_q_plot'], self.settings['num_q_plot'])
                for i_q_out in range(self.settings['num_q_plot']):
                    for i_q_in in range(self.settings['num_q_plot']):
                        ax[i_q_out, i_q_in].plot(np.real(Qq_sample[i_q_out, i_q_in, :]), np.imag(Qq_sample[i_q_out, i_q_in, :]))
                        ax[i_q_out, i_q_in].plot(np.real(Qq_roger[i_q_out, i_q_in, :]), np.imag(Qq_roger[i_q_out, i_q_in, :]))
                    
            ax[0, 0].legend(["Sampled", "RFA"])
            plt.show()      

        # Save to case data
        self.data.rfa = out_dict
        return self.data