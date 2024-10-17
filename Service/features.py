import numpy as np
import librosa
import cv2

import cfg

def extract_frequency_domain_features(audio, sr):
    # Apply FFT to convert to the frequency domain
    fft = np.fft.fft(audio, n=sr)  # Compute the FFT
    fft_magnitude = np.abs(fft[:sr // 2])  # Take magnitude of the FFT and only positive frequencies       
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    # spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    # mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    return np.hstack([ spectral_bandwidth, fft_magnitude])


def extract_time_domain_features(audio, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))   
    rms = np.mean(librosa.feature.rms(y=audio))    
    peak_amp = np.max(np.abs(audio))    
    mean_amp = np.mean(audio)
        
    return np.array([zcr, rms, peak_amp, mean_amp])


def plot_wav_to_opencv_image(data):

    # Check if the audio is stereo or mono (multi-channel)
    if len(data.shape) == 2:
        data = data[:, 0]  # If stereo, select only one channel

 
    # Use absolute value of the waveform data
    # data = np.abs(data)

    # Create a time array based on the sample rate and length of the data
    duration = data.shape[0] / features.SAMPLING_RATE
    time = np.linspace(0., duration, data.shape[0])

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, color=(78 / 255, 154 / 255, 6 / 255))
    
    # Hide axes, ticks, and labels
    plt.axis('off')  # Turn off the axis
    plt.xlim([0, duration])  # Set the x-limits
    plt.ylim([-50000, 50000])  # Set the x-limits


    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Retrieve the PNG image data from the buffer
    buf.seek(0)
    png_data = buf.getvalue()
    
    # Convert PNG binary data to a numpy array
    np_img = np.frombuffer(png_data, dtype=np.uint8)

    # Decode image from numpy array to OpenCV format (BGR)
    cv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    
    return cv_img

def find_area_and_perimeter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = 0
    total_perimeter = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)  # True indicates that the contour is closed
        
        total_area += area
        total_perimeter += perimeter

    return total_area, total_perimeter

def find_point(contours, image):
    largest_y = -1
    largest_point = None
    least_y = float('inf')
    least_point = None

    for contour in contours:
        for point in contour:
            y = point[0][1]
            if y > largest_y:
                largest_y = y
                largest_point = tuple(point[0])
            if y < least_y:
                least_y = y
                least_point = tuple(point[0])

    if largest_point is not None:
        cv2.circle(image, largest_point, 5, (0, 0, 255), 1)
    if least_point is not None:
        cv2.circle(image, least_point, 5, (0, 0, 255), 1)

    return largest_point, least_point

def pos_filter(image):
    
    
    if image is None:
        logging.error(f"Error: Image {filename} not found.")
        return False

    # Proceed with image processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray == 114
    mask = mask.astype(np.uint8) * 255
    filtered_image = cv2.bitwise_and(image, image, mask=mask)
    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image_copy = filtered_image.copy()
    largest_point, least_point = find_point(contours, filtered_image_copy)                      



    # Ensure points are found before calculation
    if largest_point is None or least_point is None:
        logging.warning(f"No valid points found in image {filename}.")
        return False    
    
    
    x_min = min(largest_point[0] , least_point[0])
    y_min = min(largest_point[1] , least_point[1])

    h_peek = largest_point[1] - least_point[1]
    w_peek = abs(largest_point[0] - least_point[0])

    x, y, w, h = x_min,  y_min, 100, h_peek
    
    total_area = 0
    total_perimeter = 0

    if (w != 0):                
        roi = filtered_image[y:y + h, x:x + w]
        total_area , total_perimeter = find_area_and_perimeter(roi)
    

    result = False
    # Determine the result based on conditions
    if (w_peek <= cfg.thres_w_peek):               
        result = True        
        
    if (h_peek < cfg.thres_h_peek):
        result = False                               

    if ((total_area < cfg.thres_total_area_min) or (total_area > cfg.thres_total_area_max)):
        result = False   
        
    return result       
                
    
    # print(f'h_peek: {h_peek} pixels')
    # print(f'w_peek: {w_peek} pixels')                
    # print(f"total_area: {total_area:.8f}")
    # print(f"total_perimeter: {total_perimeter:.8f}")
    # print(f"result: {result}")
    # print(f'------------------------------')

