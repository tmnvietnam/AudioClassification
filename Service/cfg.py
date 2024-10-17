EPOCHS = 160
BATCH_SIZE = 16
SAMPLING_RATE = 22050
LABELS =  ["ng", "ok"] 
T = 2
N = T*8

PIPE_NAME = r'\\.\pipe\TensorflowService'     

thres_w_peek = 20
thres_h_peek = 35
thres_total_area_min = 280
thres_total_area_max = 10000