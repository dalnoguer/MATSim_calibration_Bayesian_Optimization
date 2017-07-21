import xml.etree.ElementTree as ET
from utilities import *
from bayesian_optimization import *
import os
import subprocess
import pickle
import numpy as np


class CalibrationObjective(object):
    """Wrap around MatSimSimulator to create the mismatch function needed by
    BO."""

    def __init__(self, ground_truth, simulator, output_dir, metric,
                 aggregate_measure, new_config):
        """
        :param ground_truth: Dictionary containing the true aggregate measure.
        :param simulator: MatSimSimulator object needed to update the
        parameters and run the simulations.
        :param output_dir: Directory where contains the output of the
        simulation.
        :param metric: Norm to use to compute mismatch.
        :param aggregate_measure: Type of aggreagate measure used (e.g.
        daily/hourly mode shares, daily/hourly traffic counts)
        :param new_config: Name for new configuration file when updating the
        parameters.
        """
        self.ground_truth = ground_truth

        # Array form for ground truth to compute norms
        self.truth_array = np.empty(len(self.ground_truth))
        map_to_array = {}
        for i, key in enumerate(self.ground_truth):
            map_to_array[key] = i
            self.truth_array[i] = self.ground_truth[key]

        # Mapping from key to index for array form of simulation output
        self.map_to_array = map_to_array

        self.simulator = simulator
        self.output_dir = output_dir
        self.metric = metric
        self.aggregate_measure = aggregate_measure
        self.new_config = new_config

    def compute_mismatch(self, save=True, clean=True):
        """
        Compute the mismatch between truth and simulation for the specified
        aggregate measure and with the desired norm.
        :param save: bool. If True, the dictionary of the aggregate measure
        from the simulation is saved.
        :return: mismatch between truth and simulation.
        """
        if clean:
            clean_output_folder(self.output_dir)

        if self.aggregate_measure == 'daily_counts':
            raise NotImplementedError

        if self.aggregate_measure == 'hourly_counts':
            raise NotImplementedError

        if self.aggregate_measure == 'daily_shares':

            daily_shares = daily_mode_share(
                os.path.join(self.output_dir, 'output_events.xml.gz'))

            if save:
                file_name = os.path.join(self.output_dir, 'daily_shares.p')
                pickle.dump(daily_shares, open(file_name, 'wb'))

            daily_shares_array = dic2vec(daily_shares, self.map_to_array)

            # Normalize
            daily_shares_array = daily_shares_array / np.sum(daily_shares_array)

            return np.linalg.norm(daily_shares_array - self.truth_array,
                                  ord=self.metric)

        if self.aggregate_measure == 'hourly_shares':
            raise NotImplementedError

    def query_point(self, x_new):
        # Set output dir and new parameter value in new config file
        if 'outputDirectory' not in x_new:
            x_new['outputDirectory'] = self.output_dir
        self.simulator.modified_config = self.new_config
        self.simulator.update_config(x_new)
        self.simulator.run_simulation()


class MatSimSimulator(object):
    def __init__(self, base_config, modified_config, dir_matsim):
        self.base_config = base_config
        self.modified_config = modified_config
        self.dir_matsim = dir_matsim

    def update_config(self, dictionary, config=None, name=None, module=None):
        """
        Updates a configuration file for a MATsim simulation with the
        parameter values specified in the dictionary given in input.

        :param dictionary: python dictionary o values that need to be
        updated. The key is the name of the parameter.
        :param config: Configuration file that we want to update.
        :param name: name of the new file.
        :param module: module of the config file in which the parameters of
        interest can be found.
        :return:
        """
        if config is None:
            config = self.base_config

        tree = ET.parse(config)
        root = tree.getroot()

        for key, val in dictionary.items():
            if module is not None:
                tmp = root.find("module[@name={}]/param[@name={}]".format(
                    module, key))
            else:
                tmp = root.find("module/param[@name='{}']".format(key))
            try:
                tmp.set('value', val)
            except AttributeError:
                print("Attribute {} could not be set. Check "
                      "spelling.".format(key))

        if name is None:
            name = self.modified_config

        close_xml(tree, name)

    def run_simulation(self):

        if os.path.isfile(self.modified_config):

            subprocess.call(['bsub', '-W', '06:00', '-n', '1', '-R', 'rusage[mem=4000]',
                             'python', 'send_job.py', self.dir_matsim, self.modified_config])


        else:
            print('Need to specify an existing configuration file to run the '
                  'simulation')


if __name__ == '__main__':

    """
    To run simulation set the following parameters:
    :param root_dir: root directory of the project.
    :param dir_matsim: directory containing MATSim code.
    :param iterations: number of itertions performed by MATSim.
    :param ground_truth: ground truth to compute mismatch.
    :param batch_size: size of the batch size for parallel computation.
    :param span: set span of the parameters to optimize.
    :param num_iterations_BO: number of Bayesian Optimization iterations to perform.
    :param params: parameters for the kernel and the likelihood variance of the GP.
    :param x_init: array containing x coordinates of the initial data points.
    :param y_init: array containing y values corresponding to x_init.
    """

    # Define relevant directories
    root_dir = '/cluster/scratch/daln/MATsim-calibration-euler'
    base_dir = os.path.join(root_dir, 'base_files/')
    config_dir = os.path.join(root_dir, 'modified_config/')
    output_dir = os.path.join(root_dir, 'output/')

    dir_matsim = '/cluster/scratch/daln/MATsim-calibration-euler/matsim-0.8.1'

    # Define input files and overwrite config_default
    iterations = 400
    population_fraction = 1.0
    input_dictionary = {
        'inputNetworkFile': os.path.join(base_dir,
                                         'Siouxfalls_network_PT.xml'),
        'inputPlansFile': os.path.join(base_dir,
                                       'Siouxfalls_population.xml.gz'),
        'inputFacilitiesFile': os.path.join(base_dir,
                                            'Siouxfalls_facilities.xml.gz'),
        'transitScheduleFile': os.path.join(base_dir,
                                            'Siouxfalls_transitSchedule.xml'),
        'vehiclesFile': os.path.join(base_dir, 'Siouxfalls_vehicles.xml'),
        'lastIteration': str(iterations),
        'writeEventsInterval': str(iterations),
        'writePlansInterval': str(iterations),
        'flowCapacityFactor': str(population_fraction)
    }

    base_config = os.path.join(base_dir, 'config_default.xml')
    MATsim_simulator = MatSimSimulator(base_config, base_config, dir_matsim)
    MATsim_simulator.update_config(input_dictionary)

    # Initialize the Calibration objective
    ground_truth = {'car': 0.37665979,
                    'walk': 0.42293816,
                    'pt': 0.20040205}

    batch_size = 5
    mismatch_computation = {}
    for i in range(0, batch_size):
        mismatch_computation[str(i)] = CalibrationObjective(ground_truth, MATsim_simulator, output_dir, 2,
                                                            'daily_shares', config_dir)

    # Start the BO loop
    # constantPt, constantCar, constantWalk
    span = ((-0.75, 0.0), (-0.75, 0.0), (-0.75, 0.0))
    input_dim = len(span)
    num_iteartions_BO = 12

    bayesian_optimizer = batch_bayesian_optimization(interval=span, input_dim=input_dim, batch_size=batch_size)

    # Import parameters
    params = {'kernel_variance': ..., 'kernel_lengthscales': ..., 'likelihood_variance': ...}
    x_init = np.array([...])
    y_init = np.array([...])

    # Initialize model
    bayesian_optimizer.fit_model_batch(x_init,y_init,params,init = True)

    # Plot initial GP
    bayesian_optimizer.plot_GP(os.path.join(output_dir, 'plot_init')), final=False)

    for iteration_BO in range(0, num_iteartions_BO):

        # Create Output folders
        for i in range(0, batch_size):
            path = os.path.join(output_dir, 'output{}_{}'.format(str(iteration_BO), str(i)))
            os.mkdir(path)

        # Get next batch
        new_value_array = bayesian_optimizer.maximize_batch_acquisition()

        # Submit job for each new_value in new_value_array
        for i in range(0, batch_size):
            dic_new_value = {'constantPt': str(float(new_value_array[i, 0])),
                             'constantCar': str(float(new_value_array[i, 1])),
                             'constantWalk': str(float(new_value_array[i, 2]))}

            # Name for new config file
            config_name = os.path.join(config_dir, 'config{}_{}'.format(str(iteration_BO), str(i)))
            mismatch_computation[str(i)].new_config = config_name

            # Name for new output folder
            output_name = os.path.join(output_dir, 'output{}_{}/'.format(str(iteration_BO), str(i)))
            mismatch_computation[str(i)].output_dir = output_name

            mismatch_computation[str(i)].query_point(dic_new_value)

        # Compute mismatch values for each query point once job is finished
        mismatch_value_array = np.empty([batch_size, 1])
        simulation_done = False
        mismatch_computed = batch_size * [False]

        while not simulation_done:

            for i in range(0, batch_size):
                if os.path.isfile(
                        os.path.join(output_dir, 'output{}_{}/output_events.xml.gz'.format(str(iteration_BO), str(i)))) \
                        and mismatch_computed[i] == False:
                    mismatch_value_array[i, 0] = mismatch_computation[str(i)].compute_mismatch()
                    mismatch_computed[i] = True

            if all(mismatch_computed):
                simulation_done = True

        # Fit model adding last batch
        bayesian_optimizer.fit_model_batch(new_value_array, mismatch_value_array, params)

        # Plot GP
        bayesian_optimizer.plot_GP(os.path.join(output_dir, 'plot_{}'.format(str(iteration_BO))), final=False)
