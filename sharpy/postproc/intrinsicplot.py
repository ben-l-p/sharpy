import os

import numpy as np
from tvtk.api import tvtk, write_data

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings_utils
import sharpy.utils.algebra as algebra

@solver
class IntrinsicPlot(BaseSolver):
    """
    Plotter for intrinsic solution to Paraview format

    """
    solver_id = 'IntrinsicPlot'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['name_prefix'] = 'str'
    settings_default['name_prefix'] = ''
    settings_description['name_prefix'] = 'Name prefix for files'

    settings_types['stride'] = 'int'
    settings_default['stride'] = 1
    settings_description['stride'] = 'Stride length, plotting 1 in every \'stride\' timesteps'

    settings_table = settings_utils.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None

        self.folder = ''
        self.filename = ''
        self.filename_for = ''
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None, restart=False):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings_utils.to_custom_types(self.settings, self.settings_types, self.settings_default)
        # create folder for containing files if necessary
        self.folder = data.output_folder + '/intrinsic/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = (self.folder +
                         self.settings['name_prefix'] +
                         'beam_' +
                         self.data.settings['SHARPy']['case'])
        self.caller = caller

    def run(self, **kwargs):
        self.n_tstep = len(self.data.intrinsic.t)
        self.n_nodes = self.data.structure.num_node
        self.n_elem = self.data.structure.num_elem

        self.conn = np.zeros([self.n_elem, 3], dtype=int)
        self.node_id = np.zeros(self.n_nodes, dtype=int)
        self.elem_id = np.zeros(self.n_elem, dtype=int)

        for i_elem in range(self.n_elem):
            self.conn[i_elem, :] = self.data.structure.elements[i_elem].reordered_global_connectivities
            self.elem_id[i_elem] = i_elem

        for i_t in range(0, self.n_tstep, self.settings['stride']):
            self.write_beam(i_t)

            self.write_aero(i_t)

    def write_beam(self, i_t):
        i_t_filename = (self.filename + ('%06u' % i_t) + '.vtu')

        coords = self.data.intrinsic.ra[i_t, :, :].T

        if np.any(np.isnan(coords)):
            return

        ug = tvtk.UnstructuredGrid(points=coords)
        ug.set_cells(tvtk.Line().cell_type, self.conn)
        ug.cell_data.scalars = self.elem_id
        ug.cell_data.scalars.name = 'elem_id'

        write_data(ug, i_t_filename)

    def write_aero(self, i_t):
        pass
