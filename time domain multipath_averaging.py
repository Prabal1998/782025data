import re
from math import sqrt, atan2, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter1d
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.stats import rayleigh, rice ,norm, uniform
import os
import scipy.optimize as opt
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
from matplotlib.ticker import ScalarFormatter

def preprocess_csi(csi_raw, csi_stack_time_series_amplitudes,csi_stack_time_series_phases,centrl_feq,rssi,rssi_arr):
    """
    Preprocess CSI data to extract amplitude and phase.
    """
    imaginary = []
    real = []
    amplitudes = []
    phases = []


    # Create list of imaginary and real numbers from CSI
    for i in range(len(csi_raw)):
        if i % 2 == 0:
            imaginary.append(csi_raw[i])
        else:
            real.append(csi_raw[i])

    # Transform imaginary and real into amplitude and phase
    for i in range(int(len(csi_raw) / 2)):
        amplitudes.append(sqrt(imaginary[i] ** 2 + real[i] ** 2))
        phases.append(atan2(imaginary[i], real[i]))

    csi_stack_time_series_amplitudes, csi_stack_time_series_phases = csi_array_stack(amplitudes,phases,csi_stack_time_series_amplitudes,csi_stack_time_series_phases)

    #rssi_array collection
    rssi_arr.append(rssi)
    
    

    # Define channel parameters for channel 4
    B = 20e6  # Bandwidth in Hz (20 MHz)
    N = 64    # Number of subcarriers
    fc = int(centrl_feq)
    #fc = 2.412e9 # Center frequency in Hz (2.4 GHz)

    # Generate subcarrier indices for a 20 MHz channel
    num_subcarriers = len(amplitudes)  # Number of subcarriers
    delta_f = B / num_subcarriers  # Subcarrier spacing in Hz
    subcarrier_indices = np.arange(-num_subcarriers // 2, num_subcarriers // 2)
    
    size = len(subcarrier_indices)
    shape = (size,)  # Since it's a 1D array, the shape is a tuple with one element
    #print(f"subcarrier indices: {size}, shape: {shape}")


    #subcarrier_indices = np.fft.fftshift(np.arange(-num_subcarriers // 2, num_subcarriers // 2))  
    subcarrier_frequencies = (fc + subcarrier_indices * delta_f) # Subcarrier frequencies in Hz
    #left shift by 2 to adjust the position of frequencies aligning with indices
    # Shift the subcarrier indices to the left by 1 position
    #subcarrier_frequencies= np.roll(subcarrier_frequencies, -2)
    
    size = len(subcarrier_frequencies)
    shape = (size,)  # Since it's a 1D array, the shape is a tuple with one element
    #print(f"subcarrier frequencies: {size}, shape: {shape}")

    size = len(amplitudes)
    shape = (size,)  # Since it's a 1D array, the shape is a tuple with one element
    #print(f"subcarrier amplitudes: {size}, shape: {shape}")

    size = len(phases)
    shape = (size,)  # Since it's a 1D array, the shape is a tuple with one element
    #print(f"subcarrier phases: {size}, shape: {shape}")


    return amplitudes, phases, subcarrier_indices, subcarrier_frequencies,csi_stack_time_series_amplitudes, csi_stack_time_series_phases,rssi_arr

def csi_array_stack(amplitudes,phases,csi_stack_time_series_amplitudes,csi_stack_time_series_phases):
     csi_stack_time_series_amplitudes = np.concatenate((csi_stack_time_series_amplitudes,np.reshape(amplitudes,(1,len(amplitudes)))),axis=0)
     csi_stack_time_series_phases = np.concatenate((csi_stack_time_series_phases,np.reshape(phases,(1,len(phases)))),axis=0)
     
     return csi_stack_time_series_amplitudes, csi_stack_time_series_phases

def compute_idft(amplitudes, phases, apply_window=False, window_type='hamming'):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) of the CSI data.
      Optionally apply a windowing function to the amplitudes.
    """

    # Unwrap the phase values along the last axis (columns)
    phases = np.unwrap(phases, axis=1)

    #initalizing time series list
    time_domain_stack = []

    if apply_window:
        if window_type == "hamming":
            window = np.hamming(len(amplitudes))
        elif window_type == "hanning":
            window = np.hanning(len(amplitudes))
        else:
            raise ValueError("Unsupported window type. Use 'hamming' or 'hanning'.")
        
        amplitudes = amplitudes * window
    
    for i in range(amplitudes.shape[0]):
        # Convert amplitude and phase to complex frequency-domain representation
        csi_complex = [amp * np.exp(1j * phase) for amp, phase in zip(amplitudes[i], phases[i])]

        # Perform IDFT to get the time-domain multipath profile
        time_domain_csi = np.fft.ifft(csi_complex)

        # Time domain index
        #time_indices = np.arange(len(time_domain_csi))  # Time indices starting with 0 for LOS inside a room  
        #time_resolution = 1 / (2 * np.pi * 20e6)  # Time resolution in seconds
        #time_values = time_indices + time_resolution * 1e9  # Convert to nanoseconds
        time_values=np.zeros(10)
        
        time_domain_stack.append(time_domain_csi)
    
    time_domain_stack = np.array(time_domain_stack)
    print("time_domain_stack")
    print(np.shape(time_domain_stack))

    return time_domain_stack, time_values

def process_csi_template(phases_stack):
    """
    Process to eliminate nonlinear phase error templates.
    """

    # Unwrap phases
    phases_stack = np.unwrap(phases_stack)

    # Perform linear fit on the middle part of subcarriers

    mid_start = len(subcarrier_indices) // 4
    mid_end =  3 * len(subcarrier_indices) // 3
    mid_indices = subcarrier_indices[mid_start:mid_end]
    mid_phases = phases[mid_start:mid_end]

    # Linear fit
    fit_coeffs = np.polyfit(mid_indices, mid_phases, 1)
    linear_fit = np.polyval(fit_coeffs, subcarrier_indices)

    # Subtract linear fit to obtain nonlinear phase error
    nonlinear_phase_error = phases - linear_fit

    #Calculate mean and standard deviation of nonlinear phase error
    mean_phase_error = np.mean(nonlinear_phase_error)
    std_phase_error = np.std(nonlinear_phase_error)

    return amplitudes, mean_amplitude, phases, subcarrier_frequencies, nonlinear_phase_error, mean_phase_error, std_phase_error

def csi_nonlinear_sanitization(csi_raw,nonlinear_phase_error):
    """
    Sanitize CSI data by removing nonlinear phase error.
    """
    # Preprocess CSI to get amplitude and phase
    amplitudes, phases, subcarrier_indices, subcarrier_frequencies= preprocess_csi(csi_raw)

    # Unwrap phases
    phases = np.unwrap(phases)

    #remove non-linear phase error obtained when there was non-multipath error

    # Subtract nonlinear phase error from the original phases
    sanitized_phases = phases - nonlinear_phase_error

    # Convert amplitude and sanitized phase to complex frequency-domain representation
    

    return csi_sanitized

def agc_effect(csi_raw, window_size=8):
    """
    Apply Automatic Gain Control (AGC) effect to CSI data using a moving average.
    """
    # Preprocess CSI to get amplitude and phase
    amplitudes, phases, subcarrier_indices, subcarrier_frequencies = preprocess_csi(csi_raw)

    # Compute moving average for amplitude
    amplitudes_moving_avg = np.convolve(amplitudes, np.ones(window_size) / window_size, mode='same')

    # Compute moving average for phase
    phases_moving_avg = np.convolve(phases, np.ones(window_size) / window_size, mode='same')

    # Compensate amplitude and phase using the moving average
    amplitudes_compensated = amplitudes / amplitudes_moving_avg
    phases_compensated = phases - phases_moving_avg

    return amplitudes_compensated, phases_compensated
    

def phase_sfo(phases, subcarrier_indices):
    """
    Adjust the phase for each subcarrier using the formula:
    ϕ_k := ϕ_k - k * ε
    where k is the subcarrier index and ε is a constant.
    """
    #obtained from presice power delay profiling
    epsilon=0.0755
    
    # Perform the vector operation
    adjusted_phases_sfo = phases - (subcarrier_indices * epsilon)
    
    return adjusted_phases_sfo

def central_leakage(amplitudes):
    """
    " reference https://download.ni.com/evaluation/rf/Introduction_to_WLAN_Testing.pdf" """


def csi_parse_n_preprocess_n_stack():

# This function check and correct the csi retrieved from COTS devices.
    os.chdir('C:\\Users\\jcpra\\Downloads\\Study Research Reference Development\\Study Masters KU\\Research\\Final year\\CSI\\Data collection')
    FILE_NAME = input('Enter the filename with extension: ')
    #FILE_NAME = "FILE_NAME"

    f = open(FILE_NAME)

    # Skip the first row (header)
    next(f)

    #initalize csi_stack_time_series_amplitude array of shape (1, length of amplitude obtained from raw CSI. 64 used here)
    csi_stack_time_series_amplitudes = np.zeros((1,64))
    csi_stack_time_series_phases= np.zeros((1,64))

    rssi_arr = []

    centrl_feq = input('Enter central frequency: ')
    

    for j, l in enumerate(f.readlines()):
        # Parse string to extract the CSI data
        #csi_string = re.findall(r"\[(.*)\]", l)[0]

        # Parse string to extract the CSI data / non-greedy approach
        csi_string = re.findall(r"\[(.*?)\]", l)
        if csi_string:
            csi_string = csi_string[0]
        else:
            continue

        # Debugging: Print the raw CSI string
        #print(f"Raw CSI string: {csi_string}")

        # Normalize spaces and remove leading/trailing spaces
        csi_string = re.sub(r"\s+", " ", csi_string.strip())

        # Use a custom delimiter to separate concatenated integers with '-' or '+'
        csi_string = re.sub(r"(?<=\d)-(?=\d)", " -", csi_string)  # Add a space before negative numbers concatenated with digits
        csi_string = re.sub(r"(?<=\d)\+(?=\d)", " +", csi_string)  # Add a space before positive numbers concatenated with digits

        # Replace every occurrence of '00' with '0 0'
        csi_string = re.sub(r"\b00\b", "0 0", csi_string)

        # Extract only valid integers
        csi_raw = re.findall(r"-?\d+", csi_string)

        # Convert to integers
        csi_raw = [int(x) for x in csi_raw]

        #for RSSI
        # Using regex to find the value between the 3rd and 4th delimiter (comma)
        # Corrected regex to find the value between the 3rd and 4th delimiter
        
        #match = re.search(r"^(?:[^,]*,){3}([^,]*)", l)

        #if match:
         #   rssi_string = match.group(1)
          #  print(rssi_string)  # Outputs: -56
        #else:
        #    print("No match found")
        
        rssi_string = re.findall(r"(?:[^,]*,[^,]*,[^,]*,)([^,]*)", l)[0]
        rssi = int(rssi_string)

        # Validate the count of integers
        if len(csi_raw) != 128:
            print(f"Error: Expected 128 integers, but found {len(csi_raw)}")
            #print(f"Delimited string: {csi_raw}")
            continue  # Skip this line and move to the next

        # Return the size and shape of the delimited string
        #size = len(csi_raw)
        #shape = (size,)  # Since it's a 1D array, the shape is a tuple with one element
        #print(f"Delimited string size: {size}, shape: {shape}")

        amplitudes, phases, subcarrier_indices, subcarrier_frequencies,csi_stack_time_series_amplitudes, csi_stack_time_series_phases, rssi_arr = preprocess_csi(csi_raw,csi_stack_time_series_amplitudes,csi_stack_time_series_phases,centrl_feq,rssi,rssi_arr)
        
        #nullifying ESP32 hardware related error in subcarrier 0 and 1.
        csi_stack_time_series_amplitudes_null_hrwr_remvl = csi_stack_time_series_amplitudes
        csi_stack_time_series_amplitudes_null_hrwr_remvl[:,:2] = 0

        #nullifying ESP32 hardware related error in subcarrier 0 and 1.
        csi_stack_time_series_phases_null_hrwr_remvl = csi_stack_time_series_phases
        csi_stack_time_series_phases_null_hrwr_remvl[:,:2] = 0

#in concatenate after initalization with zeros one rows adds up removing that row
    return amplitudes, phases, subcarrier_indices, subcarrier_frequencies,csi_stack_time_series_amplitudes[1:,:], csi_stack_time_series_phases[1:,:],csi_stack_time_series_amplitudes_null_hrwr_remvl[1:,:], csi_stack_time_series_phases_null_hrwr_remvl[1:,:],centrl_feq, rssi_arr

def moving_avg_row_column(csi_stack_time_series_amplitudes,csi_stack_time_series_phases):
    
    # Compute averages
    row_avg_amp= np.mean(csi_stack_time_series_amplitudes, axis = 0, keepdims = True)  #  average across columns / horizontal
    col_avg_amp= np.mean(csi_stack_time_series_amplitudes, axis = 0, keepdims = True)  #  average across rows/ vertical
    print("Shape of vertical average amplitude")
    print(np.shape(row_avg_amp))
    print(np.shape(col_avg_amp))

        # Compute averages
    row_avg_phases = np.mean(csi_stack_time_series_phases,axis = 0, keepdims = True)  #  average across columns / horizontal
    col_avg_phases= np.mean(csi_stack_time_series_phases, axis = 0, keepdims = True )  # average across rows/ vertical

    #Remove average gain effect
    #col_avg_amp_gain_rmbl = col_avg_amp / np.mean(gain_arr)
    #col_avg_phases_gain_rmbl = col_avg_phases / np.mean(gain_arr)

   
    return row_avg_amp, col_avg_amp, row_avg_phases, col_avg_phases#, col_avg_amp_gain_rmbl, col_avg_phases_gain_rmbl

def distribution_plot(amplitude_centrl_avg_null_drop, phases_centrl_avg_null_drop):
    #convert into 1D array

    data = amplitude_centrl_avg_null_drop.flatten()

    # The 'fit' method returns parameters estimated by maximum likelihood.
    #params = norm.fit(data)

    # This returns a tuple (loc, scale). For the Rayleigh distribution, the 'loc'
    # parameter is often near 0, and the 'scale' parameter is the key parameter.
    #print("Estimated parameters (loc, scale):", params)

    # Step 3: Generate x-values and compute PDF values for plotting.
    x_vals = np.linspace(data.min(), data.max(), 100)
    #pdf_vals = norm.pdf(x_vals, *params)


    # Rician distribution fit
    params = rice.fit(data, floc = 0)
    print("Estimated parameters (x/sigma (line of sight to scatter), loc, scale (noise or such)):", params)
    pdf_vals = rice.pdf(x_vals, *params )


    #KS Test for Rician distribution
    b_value = 2.0    # shape parameter
    scale = 2.0      # scale parameter (σ)
    loc = 0          # typically the location is 0 for a Rician distribution
    ks_stat_norm, p_value_norm = stats.kstest(data, 'rice', args=(b_value, loc, scale))
    print(f"KS Test for Rician Distribution in case of CSI amplitude: Statistic={ks_stat_norm:.4f}, p-value={p_value_norm:.4f}")


    # Step 4: Plot the fitted PDF along with a histogram of the data.
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, density=True, alpha=0.6, color='gray', edgecolor='black', 
             label="Data Histogram")
    plt.plot(x_vals, pdf_vals, label=" Rice Fitted  PDF", color="green", linewidth=2)
    plt.xlabel("Value")
    plt.ylabel("Probability density")
    plt.title("Fitting a Rice Distribution to Data")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot histogram
    sns.histplot(data, bins=50, kde=True)

    # Labels and title
    plt.xlabel("CSI amplitude")
    plt.ylabel("Frequency")
    plt.title("Histogram and distribution of CSI amplitude")

    # Show plot
    plt.show()





    

    data = phases_centrl_avg_null_drop.flatten()


    
    # The 'fit' method returns parameters estimated by maximum likelihood.
    params = uniform.fit(data)

    # This returns a tuple (loc, scale). For the Rayleigh distribution, the 'loc'
    # parameter is often near 0, and the 'scale' parameter is the key parameter.
    #print("Estimated parameters (loc, scale):", params)

    # Step 3: Generate x-values and compute PDF values for plotting.
    x_vals = np.linspace(data.min(), data.max(), 100)

    pdf_vals = uniform.pdf(x_vals, *params)

    # KS Test for uniform Distribution
    ks_stat_norm, p_value_norm = stats.kstest(data, 'uniform', args=(np.mean(data), np.std(data)))
    print(f"KS Test for Uniform Distribution in case of CSI amplitude: Statistic={ks_stat_norm:.4f}, p-value={p_value_norm:.4f}")




    # Plot histogram
    sns.histplot(data, bins=50, kde=True)

    # Labels and title
    plt.xlabel("Phases")
    plt.ylabel("Frequency")
    plt.title("Histogram and distribution of CSI phases")

    # Show plot
    plt.show() 


def abnormality_removal_csi(amplitude,phase,subcarrier_indices):
    #dropout of null carrier in amplitude
    
    amplitude_centrl_avg = amplitude
    amplitude_centrl_avg[:,32] = (amplitude_centrl_avg[:,31] + amplitude_centrl_avg[:,33]) / 2 
    amplitude_centrl_avg_null_drop = amplitude_centrl_avg[:,6:-5]

    
    phases_centrl_avg = phase
    phases_centrl_avg[:,32] = (phases_centrl_avg[:,31] + phases_centrl_avg[:,33]) / 2 
    phases_centrl_avg_null_drop = phases_centrl_avg[:,6:-5]


    #adding null or padding zero at first six column and last five column to perform IFFT

    amplitude_centrl_avg_null_pad = np.hstack((np.zeros((amplitude_centrl_avg_null_drop.shape[0],6)),amplitude_centrl_avg_null_drop,np.zeros((amplitude_centrl_avg_null_drop.shape[0],5))))
    
    phases_centrl_avg_null_pad = np.hstack((np.zeros((phases_centrl_avg_null_drop.shape[0],6)),phases_centrl_avg_null_drop,np.zeros((phases_centrl_avg_null_drop.shape[0],5))))

        # Phase correction for columns 6 to 19 (indices 6–19 inclusive → 14 elements) //from experiment
    correction_6_19 = np.array([
    0.259465564, 0.333715803, 0.285353258, 0.286522599, 0.238665564,
    0.138833807, 0.102715827, 0.020743936, 0.041853653, 0.040637696,
    -0.021876513, -0.027076513, -0.064011887, -0.069211887 ])


    # Phase correction for columns 41 to 58 (indices 41–58 inclusive → 18 elements) //from experiment
    correction_41_58 = np.array([
    -0.094562304, -0.042048691, -0.104962304, -0.140456064, -0.090157559,
    -0.095357559, -0.068048691, -0.134946347, -0.02455901, -0.063966193,
    -0.069166193, 0.009991953, -0.030654187, -0.084766193, -0.05575901,
    -0.095166193, -0.152557559, -0.07135901 ])

    #defining and initalizing phase correction array

    phases_centrl_avg_null_pad_correx = phases_centrl_avg_null_pad 


    # Apply correction to subcarrier indices 6 to 19
    phases_centrl_avg_null_pad_correx[:, 6:20] -= correction_6_19

    # Apply correction to subcarrier indices 41 to 58
    phases_centrl_avg_null_pad_correx[:, 41:59] -= correction_41_58

    phases_centrl_avg_null_pad_non_linear_rmvl = np.zeros((phases_centrl_avg_null_pad_correx.shape[0], phases_centrl_avg_null_pad_correx.shape[1]))
    amplitude_centrl_avg_null_pad_non_linear_rmvl = np.zeros((amplitude_centrl_avg_null_pad.shape[0],amplitude_centrl_avg_null_pad.shape[1]))

    amplitude_centrl_avg_null_pad_non_linear_rmvl_temp = np.zeros((amplitude_centrl_avg_null_pad.shape[0],amplitude_centrl_avg_null_pad.shape[1]))

    # Extrapolation


    for i in range(len(phases_centrl_avg_null_pad_correx)):
        phases_centrl_avg_null_pad_non_linear_rmvl [i,:] = extrapolation_end_phase(phases_centrl_avg_null_pad_correx [i,:], subcarrier_indices)

    for i in range(len(amplitude_centrl_avg_null_pad)):
        #amplitude_centrl_avg_null_pad_non_linear_rmvl_temp [i,:] = medfilt(amplitude_centrl_avg_null_pad[i,:], kernel_size = 5)
        #amplitude_centrl_avg_null_pad_non_linear_rmvl_temp[i,:] = uniform_filter1d(amplitude_centrl_avg_null_pad_non_linear_rmvl_temp[i,:], size = 5)
        amplitude_centrl_avg_null_pad_non_linear_rmvl[i,:] = extrapolation_end_amp(amplitude_centrl_avg_null_pad[i,:],amplitude_centrl_avg_null_pad_non_linear_rmvl_temp[i,:], subcarrier_indices)
        #amplitude_centrl_avg_null_pad_non_linear_rmvl[i,:] = extrapolation_end_amp(amplitude_centrl_avg_null_pad[i,:],amplitude_centrl_avg_null_pad[i,:], subcarrier_indices)
    
    #Sfo removal and filtering
    for i in range(len(phases_centrl_avg_null_pad_correx)):
         phases_centrl_avg_null_pad_non_linear_rmvl [i,:] = phase_sfo (phases_centrl_avg_null_pad_non_linear_rmvl[i,:], subcarrier_indices)
         #uniform filtering size 5 and median filtering size 5
         phases_centrl_avg_null_pad_non_linear_rmvl [i,:] = medfilt(phases_centrl_avg_null_pad_non_linear_rmvl[i,:], kernel_size = 5)
         phases_centrl_avg_null_pad_non_linear_rmvl [i,:] = uniform_filter1d(phases_centrl_avg_null_pad_non_linear_rmvl[i,:], size = 5)

    #
    #phases_centrl_avg_null_pad_non_linear_rmvl[:,59:64] = 0
    #phases_centrl_avg_null_pad_non_linear_rmvl[:,0:6] = 0

    #amplitude null carrier zeros
    #amplitude_centrl_avg_null_pad_non_linear_rmvl[:,0:6] = 0

    return amplitude_centrl_avg_null_drop, phases_centrl_avg_null_drop, amplitude_centrl_avg_null_pad, phases_centrl_avg_null_pad, phases_centrl_avg_null_pad_non_linear_rmvl, amplitude_centrl_avg_null_pad_non_linear_rmvl


def extrapolation_end_phase(amplitude_centrl_avg_null_pad,subcarrier_indices):
    # --- First extrapolation ---
    x_known_1 = subcarrier_indices[6:32]
    y_known_1 = amplitude_centrl_avg_null_pad[6:32]

# Fit a degree-2 polynomial
    coeffs_1 = np.polyfit(x_known_1, y_known_1, deg = 1)
    poly_1 = np.poly1d(coeffs_1)

    x_predict_1 = subcarrier_indices[0:6]
    y_predict_1 = poly_1(x_predict_1)

    amplitude_centrl_avg_null_pad[0:6] = y_predict_1  # overwrite extrapolated region

    # --- Second extrapolation ---
    x_known_2 = subcarrier_indices[32:59]
    y_known_2 = amplitude_centrl_avg_null_pad[32:59]

    coeffs_2 = np.polyfit(x_known_2, y_known_2, deg = 1)
    poly_2 = np.poly1d(coeffs_2)

    x_predict_2 = subcarrier_indices[59:64]
    y_predict_2 = poly_2(x_predict_2)

    amplitude_centrl_avg_null_pad[59:64] = y_predict_2
    amplitude_centrl_avg_null_pad_xtrapol = amplitude_centrl_avg_null_pad


    return amplitude_centrl_avg_null_pad_xtrapol

def extrapolation_end_amp(amplitude_centrl_avg_null_pad,amplitude_centrl_avg_null_pad_non_linear_rmvl_temp,subcarrier_indices):
    #amplitude_centrl_avg_null_pad_non_linear_rmvl_temp = amplitude_centrl_avg_null_pad
    # --- First extrapolation ---
    #x_known_1 = subcarrier_indices[6:59]
    #y_known_1 = amplitude_centrl_avg_null_pad_non_linear_rmvl_temp[6:59]



    #cubic spline implementation
    #cubic_spline_param = CubicSpline(x_known_1, y_known_1, bc_type='natural', extrapolate=True)
    #extension x points

    #x_predict_1 = subcarrier_indices[0:6]
    #y_predict_1 = cubic_spline_param(x_predict_1)
    #amplitude_centrl_avg_null_pad[0:6] = y_predict_1

    #Second extrapolation
    #x_known_1 = subcarrier_indices[6:59]
    #y_known_1 = amplitude_centrl_avg_null_pad_non_linear_rmvl_temp[6:59]



    #cubic spline implementation
    #cubic_spline_param = CubicSpline(x_known_1, y_known_1, bc_type='natural', extrapolate=True)
    #extension x points

    #x_predict_1 = subcarrier_indices[59:64]
    #y_predict_1 = cubic_spline_param(x_predict_1)
    #amplitude_centrl_avg_null_pad[59:64] = y_predict_1


    #amplitude_centrl_avg_null_pad_xtrapol = amplitude_centrl_avg_null_pad

    x_known_1 = subcarrier_indices[6:12]
    y_known_1 = amplitude_centrl_avg_null_pad[6:12]

    # Fit a degree-2 polynomial
    coeffs_1 = np.polyfit(x_known_1, y_known_1, deg = 3)
    poly_1 = np.poly1d(coeffs_1)

    x_predict_1 = subcarrier_indices[6:12]
    y_predict_1 = poly_1(x_predict_1)
    #print(np.shape(y_predict_1))
    #print(y_predict_1)
    amplitude_centrl_avg_null_pad[0:6] = y_predict_1
    #amplitude_centrl_avg_null_pad[0:6] = y_predict_1[::-1]  # mirror and overwrite extrapolated region 
    
    

    # --- Second extrapolation ---
    x_known_2 = subcarrier_indices[54:59]
    y_known_2 = amplitude_centrl_avg_null_pad[54:59]

    coeffs_2 = np.polyfit(x_known_2, y_known_2, deg = 3)
    poly_2 = np.poly1d(coeffs_2)

    x_predict_2 = subcarrier_indices[54:59]
    y_predict_2 = poly_2(x_predict_2)
    #print(np.shape(y_predict_2))
    #print(y_predict_2)
    #amplitude_centrl_avg_null_pad[59:64] = y_predict_2[::-1]
    amplitude_centrl_avg_null_pad[59:64] = y_predict_2

    amplitude_centrl_avg_null_pad_xtrapol = amplitude_centrl_avg_null_pad
        



    return amplitude_centrl_avg_null_pad_xtrapol

def plot_same_band(csi_stack_time_series_amplitudes,csi_stack_time_series_phases, col_avg_amp,col_avg_phases):
    # Define colors for each row
    colors = cm.plasma(np.linspace(0.4, 0.9, csi_stack_time_series_amplitudes.shape[0]))  # Light gradient blue shades cm.Blue or yellow cm.YlOrBr or gold cm.plasma

    plt.figure(figsize=(8, 5))
    plt.stem(csi_stack_time_series_amplitudes[1])# linestyle='-',linewidth=1, color=colors[0], alpha=0.7)
    plt.xlabel("Subcarriers")
    plt.ylabel("Magnitude")
    plt.title("CSI magnitude plot")
    plt.show()
    # Plot each row with gradient colors
    for i in range(csi_stack_time_series_amplitudes.shape[0]):
        plt.plot(csi_stack_time_series_amplitudes[i], linestyle='-',linewidth=1, color=colors[i], alpha=0.7)

    # Plot moving averages with bold colors, similar to flatten and plot
    plt.plot(col_avg_amp.mean(axis=0), color='blue', linewidth=3, linestyle='--', marker='s', label=" Average CSI magnitude")
    #plt.plot(col_avg.mean(axis=1), color='green', linewidth=3, linestyle='-', marker='D', label="Column Avg (Bold)")

    # Formatting for clarity
    plt.xlabel("Subcarriers")
    plt.ylabel("CSI magnitude")
    plt.title("CSI magnitude plot channel 4")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.stem(csi_stack_time_series_phases[1])#, linestyle='-',linewidth=1, color=colors[0], alpha=0.7)
    plt.xlabel("Subcarriers")
    plt.ylabel("Phases")
    plt.title("CSI phase plot")
    plt.show()
    # Plot each row with gradient colors
    for i in range(csi_stack_time_series_phases.shape[0]):
        plt.plot(csi_stack_time_series_phases[i], linestyle='-',linewidth=1, color=colors[i], alpha=0.7)

    # Plot  averages with bold colors
    plt.plot(col_avg_phases.mean(axis=0), color='blue', linewidth=3, linestyle='--', marker='s', label=" Average CSI phase")
    #plt.plot(col_avg.mean(axis=1), color='green', linewidth=3, linestyle='-', marker='D', label="Column Avg (Bold)")

    # Formatting for clarity
    plt.xlabel("Subcarriers")
    plt.ylabel("Phases")
    plt.title("CSI phase plot channel 4")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def single_plot_magnitude(array, indices):
    #flatten 2d array
    array=array.flatten()
    # Plot  averages with bold colors
    plt.plot(array, color='blue', linewidth=3, linestyle='-', marker='s', label=" CSI magnitude after offset removal ")
    

    # Formatting for clarity
    plt.xlabel("Subcarriers")
    plt.ylabel("CSI magnitude")
    plt.title("CSI magnitude plot of WiFi channel 4")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def single_plot_phase(array, indices):
    #flatten 2d array
    array=array.flatten()
    # Plot  averages with bold colors
    plt.plot(array, color='blue', linewidth=3, linestyle='-', marker='s', label=" CSI phase after offset removal ")
    

    # Formatting for clarity
    plt.xlabel("Subcarriers")
    plt.ylabel("CSI phase")
    plt.title("CSI phase plot of WiFi channel 4")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def Export(dict_list):
    for name,array in dict_list.items():
        df=pd.DataFrame(array)
        filename=f"{name}.csv"
        df.to_csv(filename, index = False)

def Import(file_name):
    os.chdir('C:\\Users\\jcpra\\Downloads\\Study Research Reference Development\\Study Masters KU\\Research\\Final year\\CSI\\Data process')
    input(file_name)
    df=pd.read_csv(file_name) #string file_name
    #convert to array
    array = df.to_numpy()

def agc_removal(amplitude_centrl_avg_null_pad, col_avg_amp): #still to do
    gain = amplitude_centrl_avg_null_pad / col_avg_amp
    print('gain shape')
    print(np.shape(gain))

    amplitude_centrl_avg_null_pad = amplitude_centrl_avg_null_pad / gain #assumption gain is in timeseries
    plot_same_band(amplitude_centrl_avg_null_pad, csi_stack_time_series_phases, col_avg_amp,col_avg_phases)

    print('removing agc and ifft')

    time_domain_stack, time_values=compute_idft(amplitude_centrl_avg_null_pad,phases_centrl_avg_null_pad)
    distribution_plot(np.abs(time_domain_stack),phases_centrl_avg_null_drop)

def csi_efftv(subcarrier_frequencies,fc,csi_stack_time_series_amplitudes): #,distance)
    csi_eff = np.array([])
    #distance = np.array(distance)
    #csi_stack_time_series_amplitudes = csi_stack_time_series_amplitudes.to_numpy()  # Convert DataFrame to NumPy array if required during import
    #subcarrier_frequencies = subcarrier_frequencies.to_numpy()
    for i in range(csi_stack_time_series_amplitudes.shape[0]):
        csi_xgl = csi_stack_time_series_amplitudes[i]
        csi_res = csi_xgl * subcarrier_frequencies
        csi_res = np.sum(csi_res)
        csi_res = (1 / (fc * len(csi_xgl))) * csi_res
        csi_eff = np.append(csi_eff,csi_res)

    return csi_eff.reshape(1,len(csi_eff)) #, distance.reshape(1,distance.size)

    
def cse_efftv_fit_distance():

    # Given known distance measurements (meters)
    distances = np.array([1.10, 1.20, 1.30, 1.40, 1.50, 1.80, 2, 2.23, 3, 3.5, 4])  # Provide actual distance values
    CSI_eff = np.array([4.71615481310452E-09, 5.18143464412729E-09, 2.18902789599185E-09,
                         2.2834083543892E-09, 2.67003702393956E-09, 2.40600396974664E-09,
                         7.51561883158755E-10, 1.3791403758612E-09, 3.9781622200815E-10, 7.10217869735713E-10, 2.56734216006827E-10])  # Provide corresponding CSI_eff values

    # Normalize CSI_eff to avoid numerical instability
    CSI_norm = CSI_eff * (10**10) # This operation prevents memory overflow and customize the equation

    # Define the function based on FILA's propagation model ()
    def path_loss_model(distances, K):
        return K * distances**(-3/2)  # We iterate over n separately

    # Constants
    c = 3e8  # Speed of light (m/s)
    f0 = 2412000000  # Provide frequency value
    params, covariance = opt.curve_fit(path_loss_model, distances, CSI_norm)
    K_est = params[0]
    K1 = (c / f0) * (1/((4 * np.pi) ** (3/2)))
    sigma_est = ((K_est / K1) * (10**-10)) ** 2
    residuals = CSI_eff - path_loss_model(distances, K_est)
    error = np.sum(residuals ** 2)
    print(f"n = {3}: Estimated K = {K_est}, Estimated σ = {sigma_est}, Error = {error}") 

    # Reconstruct fitted curve for plotting (de-normalized)
    fitted_curve = path_loss_model(distances, K_est) * 1e-10

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(distances, CSI_eff, 'o-', label='Measured CSI_eff', color='royalblue')
    plt.plot(distances, fitted_curve, '--', label='Fitted Curve (n=3)', color='darkorange')

    # Formatting
    plt.xlabel('Distance (m)')
    plt.ylabel('CSI_eff')
    plt.title('CSI_eff vs Distance with Fitted Path-Loss Model')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    # Scientific notation for y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    plt.show()

    return sigma_est

def csi_agc_adjust_fr_distance(csi_stack_time_series_amplitudes,csi_stack_time_series_phases,rssi_arr, num_carriers):

    gain_arr = []

    #gain per bins
    gain_arr_bins = []

    csi_stack_time_series_magnitudes = np.zeros((csi_stack_time_series_amplitudes.shape[0], csi_stack_time_series_amplitudes.shape[1]))


    # using parseval's theorem to find the gain
    for i in range(len(csi_stack_time_series_amplitudes)):
        for i in range(csi_stack_time_series_amplitudes.shape[1]):
                
                #complex representation+absolute value+reshape
                abs_csi_stack_time_series_temp = np.abs([amp * np.exp(1j * phase) for amp, phase in zip(csi_stack_time_series_amplitudes[i], csi_stack_time_series_phases[i])]).reshape(1,csi_stack_time_series_amplitudes.shape[1])
        csi_stack_time_series_magnitudes[i,:] = abs_csi_stack_time_series_temp
                
        power_freq = np.sum(csi_stack_time_series_magnitudes[i]) / np.sqrt(csi_stack_time_series_amplitudes.shape[1])
        power_time = rssi_arr[i]
        power_time_watt = (10 ** (power_time / 10) ) / 1000 #dBm to watt
        gain = power_freq / power_time_watt
        gain_arr.append(gain)
        #
        #gain_arr_bins.append(gain_arr_bin)
    
    gain_arr_bins = [float(x / num_carriers) for x in gain_arr]

    return gain_arr, gain_arr_bins
       


def csi_agc_adjust_stack(csi_stack_time_series_amplitudes,gain_arr_bins):

    csi_stack_time_series_amplitudes_agc_adjst = np.zeros((csi_stack_time_series_amplitudes.shape[0],csi_stack_time_series_amplitudes.shape[1]))

    for i in range(csi_stack_time_series_amplitudes.shape[0]):
            csi_stack_time_series_amplitudes_agc_adjst[i,:] = csi_stack_time_series_amplitudes[i] / gain_arr_bins[i]
    return csi_stack_time_series_amplitudes_agc_adjst

def calculate_distance(csi_eff, sigma_est):
    # Calculate distance (d) using the given formula.
    c = 3e8  # Speed of light (m/s)
    CSI_eff = csi_eff
    #CSI_eff = float(input('CSI_eff: '))
    CSI_norm = CSI_eff * (10**10)
    fi = int(input('Central Carrier: '))
    n = int(input('No. of fading path loss component: '))
    sigma = sigma_est
    # Compute the expression inside the brackets
    temp_calc = (((c / (fi * CSI_norm))* 10**10) ** 2) * (sigma)
    # Compute distance d
    distance = ((1 / (4 * np.pi)) * (temp_calc ** (1/n)))
    return distance


def lse_trilateration_eqn():
    # Example input (replace with actual values)
    x1 = int(input('x1: '))
    x2 = int(input('x2: '))
    y1 = int(input('y1: '))
    y2 = int(input('y2: '))
    x3 = int(input('x3: '))
    y3 = int(input('y3: '))
    d1 = float(input('d1 distance: '))
    d2 = float(input('d2 distance: '))
    d3 = float(input('d3 distance: '))
    rij = float(input('rij distance: '))
    rik = float(input('rik distance: '))




    A = np.array([[x2 - x1, y2 - y1],
                  [x3 - x1, y3 - y1]]) # Add more rows as needed

    b21 = float(0.5 * ((d2**2) - (d3**2) + (rij**2)))
    b31 = float(0.5 * ((d2**2) - (d1**2) + (rik)**2))

    b = np.array([b21, b31])  # Adjust based on the number of entries

    return np.linalg.inv(A.T @ A) @ A.T @ b


if __name__ == "__main__":

    """
    This script preprocesses CSI data, computes the IDFT, and plots the multipath delay profile.
    """
    #retrives the correctly parsed CSI data

    #parses and preprocess raw CSI
    amplitudes, phases, subcarrier_indices, subcarrier_frequencies,csi_stack_time_series_amplitudes, csi_stack_time_series_phases,csi_stack_time_series_amplitudes_null_hrwr_remvl, csi_stack_time_series_phases_null_hrwr_remvl,centrl_feq, rssi_arr = csi_parse_n_preprocess_n_stack()
    #nonlinear abnormality remove
    amplitude_centrl_avg_null_drop, phases_centrl_avg_null_drop, amplitude_centrl_avg_null_pad, phases_centrl_avg_null_pad, phases_centrl_avg_null_pad_non_linear_rmvl, amplitude_centrl_avg_null_pad_non_linear_rmvl = abnormality_removal_csi(csi_stack_time_series_amplitudes_null_hrwr_remvl, csi_stack_time_series_phases_null_hrwr_remvl, subcarrier_indices)
    

    row_avg_amp, col_avg_amp, row_avg_phases, col_avg_phases = moving_avg_row_column(amplitude_centrl_avg_null_pad_non_linear_rmvl, csi_stack_time_series_phases_null_hrwr_remvl)
    plot_same_band(amplitude_centrl_avg_null_pad_non_linear_rmvl,phases_centrl_avg_null_pad_non_linear_rmvl, col_avg_amp,col_avg_phases)

    # Adjust phases for each subcarrier
    adjusted_phases = phase_sfo(col_avg_phases, subcarrier_indices)
    #amplitude plot
    single_plot_magnitude(col_avg_amp, subcarrier_indices)
    #phase plot
    single_plot_phase(adjusted_phases, subcarrier_indices)
    #abnormal null and central carrier average 
    #amplitude_centrl_avg_null_drop, phases_centrl_avg_null_drop, amplitude_centrl_avg_null_pad, phases_centrl_avg_null_pad, phases_centrl_avg_null_pad_non_linear_rmvl, amplitude_centrl_avg_null_pad_non_linear_rmvl = abnormality_removal_csi(csi_stack_time_series_amplitudes_null_hrwr_remvl, csi_stack_time_series_phases_null_hrwr_remvl, subcarrier_indices)
    
    #plot_same_band(amplitude_centrl_avg_null_pad_non_linear_rmvl,phases_centrl_avg_null_pad_non_linear_rmvl, col_avg_amp,col_avg_phases)
    
    #distribution plot
    #distribution_plot(amplitude_centrl_avg_null_drop, phases_centrl_avg_null_drop)

    #abnormality_removal to perform ifft
    #time_domain_stack, time_values=compute_idft(amplitude_centrl_avg_null_pad,phases_centrl_avg_null_pad)
    #distribution_plot(np.real(time_domain_stack),phases_centrl_avg_null_drop)
    #distribution_plot(np.imag(time_domain_stack),phases_centrl_avg_null_drop)
   # distribution_plot(np.abs(time_domain_stack),phases_centrl_avg_null_drop)


    #csi_eff and distance
    #distance = input('Enter the distance in meters: ')
    #distance = float(distance)
    #csi_eff_list, distance = csi_efftv(subcarrier_frequencies,int(centrl_feq),csi_stack_time_series_amplitudes,distance)

    num_carriers = int(input("enter the number of sub carriers (excluding null): "))
    
    gain_arr, gain_arr_bins = csi_agc_adjust_fr_distance(amplitude_centrl_avg_null_pad_non_linear_rmvl,phases_centrl_avg_null_pad_non_linear_rmvl,rssi_arr, num_carriers)

    csi_stack_time_series_amplitudes_agc_adjst = csi_agc_adjust_stack(amplitude_centrl_avg_null_pad_non_linear_rmvl, gain_arr_bins)

    print('from here')


    row_avg_amp, col_avg_amp, row_avg_phases, col_avg_phases = moving_avg_row_column(csi_stack_time_series_amplitudes_agc_adjst, phases_centrl_avg_null_pad_non_linear_rmvl)
    plot_same_band(csi_stack_time_series_amplitudes_agc_adjst, phases_centrl_avg_null_pad_non_linear_rmvl, col_avg_amp, col_avg_phases)
    #amplitude plot
    single_plot_magnitude(col_avg_amp, subcarrier_indices)
    csi_eff_list= csi_efftv(subcarrier_frequencies,int(centrl_feq),csi_stack_time_series_amplitudes_agc_adjst)#,distance)
    # sigma estimate // offline and separate computation
    sigma_est = cse_efftv_fit_distance()
    distance = calculate_distance(csi_eff_list, sigma_est)
    #distance = np.array([])
    #distance = input('Enter export distance: ')
    #distance = float(distance)
    #distance = np.array([distance])
    #distance = distance.reshape(1,distance.size)

    
    #csi_stack_time_series_phases_agc_adjst = csi_agc_adjust_stack(csi_stack_time_series_phases,gain_arr_bins)
    #gain doesn't affect phases only CSI amplitude gets affected
    csi_stack_time_series_phases_agc_adjst = phases_centrl_avg_null_pad_non_linear_rmvl

    time_domain_stack, time_values=compute_idft(csi_stack_time_series_amplitudes_agc_adjst,csi_stack_time_series_phases_agc_adjst)

    #absolute value+ slicing to contain 32 sample (31 index)+flatten
    time_domain_stack_abs = np.abs(time_domain_stack)
    time_domain_stack_abs = time_domain_stack_abs[:,31]
    time_domain_samples = time_domain_stack_abs.flatten()

    print('amplitude plot and phases')
    distribution_plot(time_domain_stack_abs, np.imag(time_domain_stack[:,31]))
    
    print('magnitude plot ')
    distribution_plot(np.real(time_domain_stack[:,31]), np.imag(time_domain_stack[:,31]))


    #time_domain_stack, time_values = compute_idft(col_avg_amp,col_avg_phases )
    time_domain_stack_abs_avg = np.abs(time_domain_stack)
    
    
    #distribution_plot(np.real(time_domain_stack),phases_centrl_avg_null_drop)
    #distribution_plot(np.imag(time_domain_stack),phases_centrl_avg_null_drop)
    #print('again')

    #print('phase plot')
    #print ('average single plot')
    #distribution_plot(col_avg_amp,col_avg_phases)


    #sigma = cse_efftv_fit_distance()





    #export output
    dict_list={"amplitudes":amplitudes,"phases":phases,"subcarrier_indices":subcarrier_indices,"subcarrier_frequencies":subcarrier_frequencies,
    "csi_stack_time_series_amplitudes":csi_stack_time_series_amplitudes,"csi_stack_time_series_phases":csi_stack_time_series_phases,
    "csi_stack_time_series_amplitudes_null_hrwr_remvl":csi_stack_time_series_amplitudes_null_hrwr_remvl,
    "csi_stack_time_series_phases_null_hrwr_remvl":csi_stack_time_series_phases_null_hrwr_remvl,"row_avg_amp":row_avg_amp,"col_avg_amp":col_avg_amp,
    "row_avg_phases":row_avg_phases,"col_avg_phases":col_avg_phases,"adjusted_phases":adjusted_phases,"amplitude_centrl_avg_null_drop": amplitude_centrl_avg_null_drop,
    "amplitude_centrl_avg_null_pad":amplitude_centrl_avg_null_pad, "phases_centrl_avg_null_pad" : phases_centrl_avg_null_pad, 
    "phases_centrl_avg_null_drop": phases_centrl_avg_null_drop, "time_domain_stack": time_domain_stack,"csi_eff": csi_eff_list,"distance" :distance,
    "gain_arr":gain_arr,"gain_arr_bins":gain_arr_bins,"csi_stack_time_series_amplitudes_agc_adjst":csi_stack_time_series_amplitudes_agc_adjst,
    "csi_stack_time_series_phases_agc_adjst":csi_stack_time_series_phases_agc_adjst,"time_domain_stack_abs":time_domain_stack_abs,
    "time_domain_stack_abs_avg": time_domain_stack_abs_avg,"time_domain_samples":time_domain_samples,
    "phases_centrl_avg_null_pad_non_linear_rmvl": phases_centrl_avg_null_pad_non_linear_rmvl,"amplitude_centrl_avg_null_pad_non_linear_rmvl":amplitude_centrl_avg_null_pad_non_linear_rmvl }
    
    #return to directory of data_process

    os.chdir('C:\\Users\\jcpra\\Downloads\\Study Research Reference Development\\Study Masters KU\\Research\\Final year\\CSI\\Data process')


    Export(dict_list)

    #return to parent directory
    
    os.chdir('C:\\Users\\jcpra\\Downloads\\Study Research Reference Development\\Study Masters KU\\Research\\Final year\\CSI')


   













 # Compute IDFT to get time-domain multipath profile
    #time_domain_csi, time_values = compute_idft(amplitudes, adjusted_phases, apply_window=False, window_type="hamming")

 #     # Plot the results
 # plt.figure(figsize=(12, 6))

 #     # Plot amplitude and adjusted phase
 # plt.subplot(2, 1, 1)
 # plt.stem(subcarrier_frequencies, amplitudes, label="Stem: CSI Amplitude", linefmt='b-', markerfmt='bo', basefmt='k-')
 # plt.plot(subcarrier_frequencies, amplitudes, label="Line: CSI Amplitude line plot", color='k', linestyle="-")
 # plt.xlabel("Subcarrier Frequency (Hz)")
 # plt.ylabel("Amplitude")
 # plt.title(f"CSI Amplitude for Packet #{j}")
 # plt.legend()
 # plt.subplot(2, 1, 2)
 # plt.stem(subcarrier_frequencies, phases, label="Stem: Adjusted CSI Phases", linefmt='b-', markerfmt='bo', basefmt='k-')
 # plt.plot(subcarrier_frequencies, phases, label="Line: Adjusted CSI Phases", color='k', linestyle="-")
 # plt.xlabel("Subcarrier Frequency (Hz)")
 # plt.ylabel("Phase (radians)")
 # plt.title(f"Adjusted CSI Phase for Packet #{j}")
 # plt.legend()
 # plt.tight_layout()
 # plt.show()
 # # Plot multipath delay profile
 # plt.figure(figsize=(10, 5))
 # plt.stem(time_values, np.abs(time_domain_csi), label="Multipath Delay Profile", linefmt='b-', markerfmt='bo', basefmt='k-')
 # plt.plot(time_values, np.abs(time_domain_csi), label="Multipath Delay Profile", color='k', linestyle="-")
 # plt.xlabel("Path Index")
 # plt.ylabel("Amplitude")
 # plt.title(f"Multipath Delay Profile for Packet #{j}")
 # plt.grid()
 # plt.show()
