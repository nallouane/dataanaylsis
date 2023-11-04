#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
x64105na

Final assignment

For this project, we want to find the half-lives and decay constants of
79Sr and 79Rb,two radioactive isotopes, from data collected from detectors.
The data is provided in two CSV files. The following program attempts to read
in and combine the data, perform a minimised chi-squared fit on
the data, calculate the decay constants and half-lives, produce a useful plot
of the results, and gives the uncertainty on the decay constants and
half-lives.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin


def reading_in_and_cleaning(file1, file2):
    """
    Takes data from two csv files, reads it in
    combines it, and then removes all the data points that
    have errors or zeros in any of the lines.

    Parameters
    ----------
    File1 : csv file
    File2 : csv file

    Returns
    -------
    data_sorted : float
    returns a numpy array.

    """

    file1 = np.genfromtxt(file1, delimiter=',', dtype='float32')
    file2 = np.genfromtxt(file2, delimiter=',', dtype='float32')

    data_compiled = np.vstack((file1, file2))

    data_no_errors = data_compiled[~np.isnan(data_compiled).any(axis=1)]

    data_no_zero = np.asarray([i for i in data_no_errors if 0 not in i])

    data_no_errors = data_no_zero[~np.isnan(data_no_zero).any(axis=1)]

    data_sorted = data_no_errors[data_no_errors[:, 0].argsort()]

  # file1.close()

  # file2.close()
    return data_sorted

def convert_to_seconds(data):
    """
    Converts the first column of data to seconds from hours.

    Parameters
    ----------
    data : Float
        numpy array of floats.

    Returns
    -------
    data : Float
        Numpy array of floats.

    """
    data[:, 0] *= 3600
    return data

def identify_removing_outliers(data):
    """
    For a given data set this function identifies the outliers
    using previous known paramteres of lambda_sr and lambda_rb, data points
    which are a deviatiion of greater than 3 from this fit will be removed.

    Parameters
    ----------
    data : Float
        Numpy array of floats.

    Returns
    -------
    clean_data : Float
        Numpy array of floats.

    """

    cleaned_data = []
    outliers = []
    for row in data:
        time = row[0]
        activity = row[1]
        error = row[2]

        displacment = (model_activity(time, 0.0005, 0.005) - activity)
        if np.abs(displacment) < 3* error:
            cleaned_data.append(row)
        else:
            outliers.append(row)

    cleaned_data = np.asarray(cleaned_data)
    outliers = np.asarray(outliers)

    return cleaned_data

def model_activity(time, lambda_sr, lambda_rb):
    """
    For given decay constants the model activity is given for a certain for
    each time.

    Parameters
    ----------
    time : float.
        time in seconds.
    lambda_sr : float
        Decay constant for 79Sr.
    lambda_rb : float
        Decay constant for 79Rb.

    Returns
    -------
    activity_model : float
        activity
    """
    initial_nuclei = 6.02214076 * 10 ** (17) # the number of nuclei = moles x Avagadros constant
    terra = 10**-12

    activity_model = terra * initial_nuclei * lambda_rb * lambda_sr * \
    (np.exp(-lambda_sr*time) - np.exp(-lambda_rb*time)) \
                     / (lambda_rb - lambda_sr)

    return activity_model

def chi_squared(parameters, time, activity, error):
    """
    for model parameters a given data set it gives you the the chi - squared
    value

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    time : float.
         time in seconds.
    activity : TYPE
        experimental data for activity.
    error : TYPE
        error on the experimental activity.

    Returns
    -------
    chi_squared : float
        for the given data gives the chi-squared value.

    """
    lambda_rb = parameters[0]
    lambda_sr = parameters[1]


    chi_2 = 0

    for i in enumerate(time):
        y_pred = model_activity(time[i[0]], lambda_sr, lambda_rb)

        chi_2 += np.sum(((y_pred - activity[i[0]]) / error[i[0]])**2)

    return chi_2

def find_best_parameters(function, lambda_rb, lambda_sr, time, activity,
                         error):
    """
    Parameters
    ----------
    time : float.
         time in seconds.
    activity : TYPE
        experimental data for activity.
    error : TYPE
        error on the experimental activity.
    lambda_rb_start : TYPE
        DESCRIPTION.
    lambda_sr_start : TYPE
        DESCRIPTION.

    Returns
    -------
    best_params : TYPE
        DESCRIPTION.

    """
    best_parameters = fmin(function, [lambda_sr, lambda_rb], args=(time,
                                 activity, error),)

    return best_parameters

def plot(time, activity, error, chi_squared_reduced):
    """
    Plot of my data

    Parameters
    ----------
    time : float.
         time in seconds.
    activity : TYPE
        experimental data for activity.
    error : TYPE
        error on the experimental activity.
    chi_squared_reduced : int
        reduced chi squared for the data provided.

    Returns
    -------
    None.

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.errorbar(time, activity, yerr=error, fmt="o", color="orange",label="", ecolor='red',
            elinewidth=2, capsize=4, capthick=2, barsabove=True)
    ax.plot(time, activity)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Activity (TBq)")
    ax.set_title(f"79Rb Activity Against Time, Chi Squared Reduced \
= { chi_squared_reduced:.2f}")
    ax.grid()
    ax.legend(['Activity','error'],loc="upper right")
    plt.savefig('79Rb_Activity_against_time.png',dpi=300)
    plt.show()

def mesh_arrays(x_array, y_array):
    """
    returns two meshed arrays of size len(x_array)
    by len(y_array)

    Parameters
    ----------
    x_array : float
        numpy array.
    y_array : float
        numpy array.

    Returns
    -------
    x_array_mesh : float
        numpy array.
    y_array_mesh : float
        numpy array.

    """
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in y_array:
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for dummy_element in x_array:
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh

def contour(lambda_rb, lambda_sr, time, activity, error):
    """


    Parameters
    ----------
    lambda_rb : float
        Decay constant for 79Rb.
    lambda_sr : float
        Decay constant for 79Sr.
    time : float
        DESCRIPTION.
    activity : float
        .
    error : float
        DESCRIPTION.

    Returns
    -------
    None.

    """

    lambda_rb_values = np.linspace(0.98 * lambda_rb, 1.02 * lambda_rb,100)

    lambda_sr_values = np.linspace(0.98 * lambda_sr, 1.02 * lambda_sr,100)

    lambda_rb_values_mesh, lambda_sr_values_mesh = mesh_arrays(
         lambda_rb_values, lambda_sr_values)

    figure = plt.figure(figsize=(4, 4))

    axis = figure.add_subplot(111)

    chi_squared_mesh = np.zeros_like (lambda_rb_values_mesh)

    for row in range(len(chi_squared_mesh)):
        for column in range(len(chi_squared_mesh)):
            chi_squared_mesh [row,column] = chi_squared(
                [lambda_rb_values_mesh[row,column],
             lambda_sr_values_mesh[row,column]], time, activity, error)


    figure = plt.figure(figsize=(4, 4))

    axis = figure.add_subplot(111)

    filled_contour_plot = axis.contourf(lambda_rb_values_mesh,
                                       lambda_sr_values_mesh, chi_squared_mesh)

    contour_plot = axis.contour(lambda_rb_values_mesh, lambda_sr_values_mesh,
                        chi_squared_mesh)

    axis.clabel(contour_plot, inline=1, fontsize=10, colors='w')

    axis.clabel(filled_contour_plot, inline=1, fontsize=10, colors='w')

    plt.xlabel("Decay Constant Rb (s^-1)")
    plt.ylabel("Decay Constant Sr (s^-1)")
    plt.title("Contour Plot for the decay constants")
    plt.savefig('Contour Plot.png', dpi=300)
    plt.show()


    return None

def question(lambda_sr,lambda_rb):

    question = input("would you like to know the activity at another time?\
If so type 'yes', otherwise just press enter:")

    if question == 'yes':

       activity_time = input("what time would you like (in minutes)? :")

       try:

            if float(activity_time) > 0:

                activity = model_activity(float(activity_time), lambda_sr,
                                          lambda_rb)

                print(f"The activity at a time {activity_time} minutes is\
 {activity:.3e} TBq")

            else:
                print('The initial height needs to be greater than 0 !')

       except ValueError:
            print('Please enter a number')

       except TypeError:
            print('Make sure the number is greater than 0')

       return None


def negative_model_activity(time, lambda_sr, lambda_rb):
    """
    For given decay constants, the model activity is given for each time
    multiplied by negative 1.

    Parameters
    ----------
    time : float.
        time in seconds.
    lambda_sr : float
        Decay constant for 79Sr.
    lambda_rb : float
        Decay constant for 79Rb.

    Returns
    -------
    activity_model : float
        negative of the activity
    """

    return (-1) * model_activity(time, lambda_sr, lambda_rb)

def main():
    """
    main function

    Returns
    -------
    None.

    """
    FILE1 = "/Users/noahallouane/Downloads/Nuclear_data_1.csv"
    FILE2 = "/Users/noahallouane/Downloads/Nuclear_data_2.csv"
    dataset = reading_in_and_cleaning(FILE1, FILE2)
    dataset_in_seconds = convert_to_seconds(dataset)
    dataset_no_outliers = identify_removing_outliers(dataset_in_seconds)

    lambda_rb_start = 0.005
    lambda_sr_start = 0.0005
    best_parameters = find_best_parameters(chi_squared, lambda_rb_start,
    lambda_sr_start, dataset_no_outliers[:,0],dataset_no_outliers[:,1],
    dataset_no_outliers[:,2])

    lambda_rb = best_parameters[0]
    lambda_sr = best_parameters[1]
    contour(lambda_rb, lambda_sr, dataset_no_outliers[:,0],
            dataset_no_outliers[:,1], dataset_no_outliers[:,2])
    t_half_rb = np.log(2) / (60 * lambda_rb)
    t_half_sr = np.log(2) / (60 * lambda_sr)
    degrees_of_freedom = len(dataset_no_outliers) - 2
    chi_squared_reduced = chi_squared(best_parameters,dataset_no_outliers[:,0],
    dataset_no_outliers[:,1], dataset_no_outliers[:,2])/degrees_of_freedom
    activity_90_minutes = model_activity((90 * 60), lambda_sr, lambda_rb)

    max_activity_time = fmin(negative_model_activity,500, args=(lambda_sr,
                                                                 lambda_rb))

    max_activity = model_activity(max_activity_time[0], lambda_sr, lambda_rb)

    print(f"The maximum activity is {max_activity:.3g} Tbq \
at {max_activity_time[0]:.3g}s !")
    print(f"degrees of freedom = {degrees_of_freedom}")
    print(f"The decay constant for 79Rb is {lambda_rb:.3g} s^-1")
    print(f"The decay constant for 79Sr is { lambda_sr:.3g} s^-1")
    print(f"The half life of 79Rb is { t_half_rb:.3g} minutes")
    print(f"The half life of 79Sr is { t_half_sr:.3g} minutes")
    print(f"This plot has a reduced chi squared of {chi_squared_reduced:.2f}")
    print(f"The activity of the substance at 90 minutes \
is { activity_90_minutes:.3g} TBq")

    plot(dataset_no_outliers[:, 0], model_activity(dataset_no_outliers[:, 0],
            lambda_rb, lambda_sr), dataset_no_outliers[:, 2], chi_squared_reduced)

    question(lambda_sr,lambda_rb)


# if __name__ == "__main__":
main()
