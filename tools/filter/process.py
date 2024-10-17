import os
import cv2
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import logging

thres_w_peek = 20
thres_h_peek = 35
thres_total_area_min = 280
thres_total_area_max = 10000

thres_total_perimeter = 900

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='image_processing.log', filemode='w')

def plot_wav_to_png(wav_filename, output_png):
    # Read the wav file (returns sample rate and data)
    sample_rate, data = wavfile.read(wav_filename)

    # Check if the audio is stereo or mono (multi-channel)
    if len(data.shape) == 2:
        data = data[:, 0]  # If stereo, select only one channel

    # Create a time array based on the sample rate and length of the data
    duration = data.shape[0] / sample_rate
    time = np.linspace(0., duration, data.shape[0])

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, data, color=(78 / 255, 154 / 255, 6 / 255))
    
    # Hide axes, ticks, and labels
    plt.axis('off')  # Turn off the axis
    plt.xlim([0, duration])  # Set the x-limits
    plt.ylim([-50000, 50000])  # Set the x-limits


    # Save the plot as a PNG file
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0)  # Save without padding
    plt.close()

    print(f"Waveform saved as {output_png}")

def enumerate_wav_files_and_plot(input_folder, output_folder):
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Enumerate all .wav files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            wav_path = os.path.join(input_folder, filename)
            output_png = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            plot_wav_to_png(wav_path, output_png)


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

    # List all files in the specified folder
def process_graph_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  # Only process .png files
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Error: Image {filename} not found.")
                continue

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
                continue    
            
            # cv2.imshow(f'Peek Points {filename}', filtered_image_copy)              
  
            
            x_min = min(largest_point[0] , least_point[0])
            y_min = min(largest_point[1] , least_point[1])

            h_peek = largest_point[1] - least_point[1]
            w_peek = abs(largest_point[0] - least_point[0])

            # x, y, w, h = x_min, y_min, w_peek, h_peek            
            x, y, w, h = x_min,  y_min, min(300, image.shape[1]-x_min), h_peek
            
            total_area = 0
            total_perimeter = 0
            roi_area = 0            
            ratio_area = 0
            if (w != 0):                
                roi = filtered_image[y:y + h, x:x + w]
                roi_area = h * w
                total_area , total_perimeter = find_area_and_perimeter(roi)                
                ratio_area = total_area / roi_area
           

            result = False
            # Determine the result based on conditions
            if (w_peek <= thres_w_peek):               
                result = True        
                
            if (h_peek < thres_h_peek):
                result = False                                   

            if ((total_area < thres_total_area_min) or (total_area > thres_total_area_max)):
                result = False        
            # if (ratio_area < 0.1):                
            #     if (total_area > 3500):
            #         result = False        
            
            
  
          
            logging.info(f"file: {folder_path}/{filename}")            # Log metrics
            logging.info(f'h_peek: {h_peek} pixels')
            logging.info(f'w_peek: {w_peek} pixels')                
            logging.info(f"total_area: {total_area:.2f}")
            logging.info(f"ratio_area: {ratio_area:.2f}")
            logging.info(f"total_perimeter: {total_perimeter:.2f}")
            logging.info(f"result: {result}")
            logging.info(f'------------------------------')

        # break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    
    sub_folder_name = 'ng3'
    input_folder = f'wav/{sub_folder_name}'  # Path to your folder containing .wav files
    output_folder = f'graph/{sub_folder_name}'  # Output folder for PNG files
    enumerate_wav_files_and_plot(input_folder, output_folder)

    

    process_graph_images(f'graph/{sub_folder_name}')
    

if __name__ == '__main__':
    main()
