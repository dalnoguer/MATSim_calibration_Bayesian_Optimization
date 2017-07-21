import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import GPflow
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize
import scipy
import matplotlib.ticker as ticker


# Sequential Bayesian optimization class
class sequential_bayesian_optimization(object):
    def __init__(self, model=None, interval=None, input_dim=None, iteration=0):

        """
        :param model: GPflow model of the GP.
        :param interval: tuple containing the span of the parameters.
        :param input_dim: number of parameters (dimension of x).
        :param iteration: integer that keeps track of the BO iterations so far.
        :param optimum: list containing best Y and corresponding X so far.
        """
        self.model = model
        self.interval = interval
        self.input_dim = input_dim
        self.iteration = iteration
        self.optimum = []

    def mean(self, x):

        """
        Compute the mean of the GP at x.
        """
        x = np.array([x]).reshape(1, self.input_dim)
        mean, var = self.model.predict_f(x)
        return mean

    def minimizer(self):

        """
        Compute the minimizer of the GP mean. Returns location x.
        To avoid local minima it runs local optimization n_optimization_iterations times.
        """
        n_optimization_iterations = 20
        current_best = []

        # Optimize n_optimization_iterations to avoid local minimum
        for iter in range(0, n_optimization_iterations):

            x0 = np.array([np.random.uniform(self.interval[0][0], self.interval[0][1])])
            for dimension in range(1, self.input_dim):
                x_1 = np.array([np.random.uniform(self.interval[dimension][0], self.interval[dimension][1])])
                x0 = np.column_stack((x0, x_1))
            x0 = x0.reshape(1, self.input_dim)

            res = scipy.optimize.fmin_l_bfgs_b(self.mean, x0=x0, bounds=(self.interval), approx_grad=True)
            [x, f] = [res[0], res[1]]
            if len(current_best) > 0 and f < current_best[1]:
                current_best = [x, f]
            elif len(current_best) == 0:
                current_best = [x, f]

        return current_best[0]

    def compute_acquisition_LCB(self, x):

        """
        Compute LCB acquisition function.
        """

        x = np.array([x]).reshape(1, self.input_dim)
        mean, var = self.model.predict_f(x)
        return mean - 2 * np.sqrt(var)

    def maximize_acquisition(self):

        """
        Compute minimizer of the LCB acquisition function. Return location x.
        If iteration is 0 it returns a random point.
        To avoid local minima it runs local optimization n_optimization_iterations times.
        """

        if self.iteration == 0:
            x_init = np.array([np.random.uniform(self.interval[0][0], self.interval[0][1])])
            for dimension in range(1, self.input_dim):
                x_init_1 = np.array([np.random.uniform(self.interval[dimension][0], self.interval[dimension][1])])
                x_init = np.stack((x_init, x_init_1), axis=1)
            x_init = x_init.reshape(1, self.input_dim)
            return float(x_init)

        else:
            n_optimization_iterations = 20
            current_best = []

            # Optimize n_optimization_iterations to avoid local minimum
            for iter in range(0, n_optimization_iterations):

                x0 = np.array([np.random.uniform(self.interval[0][0], self.interval[0][1])])
                for dimension in range(1, self.input_dim):
                    x_1 = np.array([np.random.uniform(self.interval[dimension][0], self.interval[dimension][1])])
                    x0 = np.stack((x0, x_1), axis=1)
                x0 = x0.reshape(1, self.input_dim)

                res = scipy.optimize.fmin_l_bfgs_b(self.compute_acquisition_LCB, x0=x0, bounds=(self.interval),
                                                   approx_grad=True)
                [x, f] = [res[0], res[1]]
                if len(current_best) > 0 and f < current_best[1]:
                    current_best = [x, f]
                elif len(current_best) == 0:
                    current_best = [x, f]

            return float(current_best[0])

    def fit_model(self, x_new, y_new, params):

        """
        Fit GP to available data. If model is None it only uses x_new and y_new. If it exists a previous GP,
        it adds x_new and y_new to
        the current data points.
        Select type of kernel to be used.
        :param x_new: x coordinates of the new data points.
        :param y_new: y value corresponding to x_new.
        :param params: dictionary containing parameters for the kernel and the GP likelihood variance.
        """

        if self.model is None:
            Xnew = x_new
            Ynew = y_new

        else:
            Xnew = np.concatenate((self.model.X.value, np.array([x_new]).reshape(1, self.input_dim)), axis=0)
            Ynew = np.concatenate((self.model.Y.value, np.array([y_new]).reshape(1, 1)), axis=0)

        k = GPflow.kernels.Matern12(self.input_dim, lengthscales=params['kernel_lengthscales'],
                                    variance=params['kernel_variance'], ARD=True)
        self.model = GPflow.gpr.GPR(Xnew, Ynew, kern=k)
        self.model.likelihood.variance = params['likelihood_variance']
        self.iteration += 1

    def plot_GP(self, path, final):

        """
        Saves plots of the GP in directory specified by path.
        If GP is 1-D it plots GP mean, variance and acquisition function.
        If GP is higher dimensional, it plots iteration in x coordinates and best data point so far.
        :param path: sets the output directory.
        :param final: if final is False, plots the next batch computed previously.
        Need to tune the normalizer of the acquisition function to adjust it to the scale of the GP.
        """

        if self.iteration > 0:
            if self.input_dim == 1:

                xx = np.arange(self.interval[0][0], self.interval[0][1], 0.000001)
                xx = xx.reshape(xx.shape[0], 1)
                mean, var = self.model.predict_f(xx)
                acqu = mean - 2 * np.sqrt(var)
                acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu))) / 20
                plt.figure(figsize=(12, 6))
                plt.plot(self.model.X.value, self.model.Y.value, 'or', mew=2)
                plt.plot(xx, mean, 'b')
                plt.plot(xx, acqu_normalized, 'y')
                plt.fill_between(xx[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]), mean[:, 0] + 2 * np.sqrt(var[:, 0]),
                                 color='grey', alpha=0.5)
                plt.xlim(self.interval[0][0] - 0.0001, self.interval[0][1] + 0.0001)

                # Plot vertical lines at points no be queried next
                if not final:
                    for i in range(0, self.batch_size):
                        plt.axvline(self.current_batch[i])

            elif self.input_dim > 1:
                fig, ax = plt.subplots()
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.set_major_locator(ticker.MaxNLocator(integer=True))

                # Plot time series best mismatch so far

                y = self.optimum
                x = np.arange(1, len(y) + 1, 1)
                plt.plot(x, y, c='r', alpha=0.5, label = 'Accuracy')
                plt.xlabel('Iteration')
                plt.ylabel('Mismatch')
                plt.title('Best mismatch so far')

            # Save figure to path
            plt.savefig(path)
            
            # Save values in a text file
            x = str(np.ndarray.tolist(self.model.X.value))
            y = str(np.ndarray.tolist(self.model.Y.value))
            values = x + '\n' + y
            with open(path + '.txt', 'a') as out:
                out.write('X values' + '\n')
                out.write('Y values' + '\n')
                out.write(values + '\n')


# Batch Bayesian optimization class, inheriting from sequential
class batch_bayesian_optimization(sequential_bayesian_optimization):
    def __init__(self, model=None, interval=None, current_batch=[], \
                 input_dim=None, iteration=0, batch_size=None):

        """
        :param model: GPflow model of the GP.
        :param interval: tuple containing the span of the parameters.
        :param batch_size: size of the new batch.
        :param input_dim: number of parameters (dimension of x).
        :param iteration: integer that keeps track of the BO iterations so far.
        :param current_batch: list that keeps track of the current batch.
        :param optimum: list containing best Y and corresponding X so far.
        """
        self.model = model
        self.interval = interval
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.iteration = iteration
        self.current_batch = current_batch
        self.optimum = []

    def compute_soft_plus_acquisition_LCB(self, x, M, L):

        """
        Computes LCB acquisition function with local penalization at location x.
        :param M: M parameter for the local penalization heuristics.
        :param L: Lipschitz constant for the current GP.
        :param p: penalization parameter.
        Set kappa to be variable if we want to expand exploration phase.
        """
        p = 2
        kappa = min(6,self.iteration)
        x = np.array([x]).reshape(1, self.input_dim)
        mean, var = self.model.predict_f(x)
        phi = 1
        if len(self.current_batch) > 0:
            for n in range(0, len(self.current_batch)):
                x_prev = np.array([self.current_batch[n]]).reshape(1, self.input_dim)
                mean_n,var_n = self.model.predict_f(x_prev)
                z = (1 / np.sqrt(2 * var_n)) * (L * np.linalg.norm((x - x_prev), 2) - M + mean_n)
                phi *= 0.5 * scipy.special.erfc(-z)
        alpha = mean - kappa* np.sqrt(var)

        return (1 / phi**p) * np.log(1 + np.exp(alpha))

    def maximize_batch_acquisition(self):

        """
        Compute batch for the next Bayesian Optimization iteration using local penalization.
        If iteration is 0 it returns batch_size points at random.
        To avoid local minima it runs local optimization n_optimization_iterations times.
        """
        if self.iteration == 0:
            self.current_batch = []
            for i in range(0, self.batch_size):
                x_init = np.array([np.random.uniform(self.interval[0][0], self.interval[0][1])])
                for dimension in range(1, self.input_dim):
                    x_init_1 = np.array([np.random.uniform(self.interval[dimension][0], self.interval[dimension][1])])
                    x_init = np.column_stack((x_init, x_init_1))
                x_init = x_init.reshape(1, self.input_dim)
                self.current_batch = self.current_batch + [x_init]
            return np.asarray(self.current_batch).reshape(self.batch_size, self.input_dim)

        else:
            # Get constants for iteration
            M = min(self.model.Y.value)
            L = self.get_Lipschitz()

            n_optimization_iterations = 40
            self.current_batch = []

            for n in range(self.batch_size):

                current_best = []
                for iter_num in range(0, n_optimization_iterations):

                    x0 = np.array([np.random.uniform(self.interval[0][0], self.interval[0][1])])
                    for dimension in range(1, self.input_dim):
                        x_1 = np.array([np.random.uniform(self.interval[dimension][0], self.interval[dimension][1])])
                        x0 = np.column_stack((x0, x_1))
                    x0 = x0.reshape(1, self.input_dim)

                    res = scipy.optimize.fmin_l_bfgs_b(self.compute_soft_plus_acquisition_LCB, \
                                                       x0=x0, bounds=(self.interval), args=(M, L), approx_grad=True)
                    [x, f] = [res[0], res[1]]

                    if len(current_best) > 0 and f < current_best[1]:
                        current_best = [x, f]
                    elif len(current_best) == 0:
                        current_best = [x, f]

                self.current_batch = self.current_batch + [current_best[0]]

            return np.asarray(self.current_batch).reshape(self.batch_size, self.input_dim)

    def fit_model_batch(self, x_new, y_new, params, init=False):

        """
        Fit GP to available data. If model is None it only uses x_new and y_new. If it exists a previous GP,
        it adds x_new and y_new to
        the current data points.
        Select type of kernel to be used.
        :param x_new: x coordinates of the new data points.
        :param y_new: y value corresponding to x_new.
        :param params: dictionary containing parameters for the kernel and the GP likelihood variance.
        """
        if init is True:

            Xnew = x_new
            Ynew = y_new

        elif self.model is None:
            Xnew = x_new.reshape(self.batch_size, self.input_dim)
            Ynew = y_new.reshape(self.batch_size, 1)

        else:
            Xnew = np.concatenate((self.model.X.value, np.array([x_new]).reshape(self.batch_size, self.input_dim)),
                                  axis=0)
            Ynew = np.concatenate((self.model.Y.value, np.array([y_new]).reshape(self.batch_size, 1)), axis=0)

        k = GPflow.kernels.Matern12(self.input_dim, lengthscales=params['kernel_lengthscales'],
                                    variance=params['kernel_variance'], ARD=True)
        meanf = GPflow.mean_functions.Constant(0.2)
        self.model = GPflow.gpr.GPR(Xnew, Ynew,k,meanf)
        self.model.likelihood.variance = params['likelihood_variance']
        self.iteration += 1

        best_Y = np.amin(Ynew)
        self.optimum.append(best_Y)

    def get_Lipschitz(self):

        """
        Compute Lipschitz constant for the current BO iteration. Approximate it with the maximum norm of the gradient
        of the current GP.
        Runs optimization n_optimization_iterations to avoid local minimum.
        """
        n_optimization_iterations = 40
        current_best = []

        for iter_num in range(n_optimization_iterations):

            x0 = np.array([np.random.uniform(self.interval[0][0], self.interval[0][1])])
            for dimension in range(1, self.input_dim):
                x_1 = np.array([np.random.uniform(self.interval[dimension][0], self.interval[dimension][1])])
                x0 = np.column_stack((x0, x_1))
            x0 = x0.reshape(1, self.input_dim)

            res = scipy.optimize.fmin_l_bfgs_b(self.get_mean_gradient, x0=x0, bounds=(self.interval), approx_grad=True)

            [x, f] = [res[0], res[1]]

            if len(current_best) > 0 and f < current_best[1]:
                current_best = [x, f]
            elif len(current_best) == 0:
                current_best = [x, f]

        return -current_best[1]

    def get_mean_gradient(self, x):

        """
        Auxiliar function that computes the gradient of the mean at location x.
        """

        x = np.array([x]).reshape(1, self.input_dim)
        mean = self.model.predict_f_gradients(x)
        norm_grad = np.linalg.norm(mean[0])

        return -1 * norm_grad
