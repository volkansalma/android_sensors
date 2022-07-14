from threading import Thread
import threading
from time import sleep
import websocket
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import pygame as pg

#max acc read rate 2ms, 500Hz
class MobileSensorReceiver:
    def __init__(self):
        self.Acc_prev = np.zeros((1, 3), dtype = np.float64)
        self.Vel_prev = np.zeros((1, 3), dtype = np.float64)
        self.Pos_prev = np.zeros((1, 3), dtype = np.float64)

        self.Acc_curr = np.zeros((1, 3), dtype = np.float64)
        self.Vel_curr = np.zeros((1, 3), dtype = np.float64)
        self.Pos_curr = np.zeros((1, 3), dtype = np.float64)

        self.Acc_sensor_bias = np.zeros((1, 3), dtype = np.float64)
        self.Acc_sensor_stddev = np.zeros((1, 3), dtype = np.float64)
        self.Acc_cal_buffer = np.zeros((5000, 4), dtype = np.float64)
        self.Acc_median7_filter_buffer = np.zeros((7, 3), dtype = np.float64)

        self.ws = websocket.WebSocketApp("ws://192.168.41.17:8081/sensor/connect?type=android.sensor.accelerometer",
                                            on_open=self.__on_acc_open,
                                            on_message=self.__on_acc_message,
                                            on_error=self.__on_acc_error,
                                            on_close=self.__on_acc_close)

        self.prev_acc_update_timestamp = 0
        self.prev_vel_update_timestamp = 0
        self.prev_pos_update_timestamp = 0
        self.acc_cal_data_cnt = 0
        self.is_acc_calibrated = False
        self.nanosecs2secs_coef = 1. / 1000000000
        self.print_data_cnt = 0

        self.debug_raw_data = np.zeros((500,3), dtype=np.float64)
        self.debug_curr_data = np.zeros((500, 3), dtype=np.float64)
        self.debug_median_filtered_data = np.zeros((500, 3), dtype=np.float64)

    def start(self):
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def acc_median7_filter_init(self):
        #fills the last 7 data from the calibration to initialize the filter
        #data in the calibration buffer is median filtered raw data, so sensor bias needs to be substracted
        self.Acc_median7_filter_buffer = self.Acc_cal_buffer[-8:-1, 0:3] - self.Acc_sensor_bias

    def acc_median7_filter(self, AccXYZ_curr):
        #roll the data over rows, move the first data to the end of the array 
        self.Acc_median7_filter_buffer = np.roll(self.Acc_median7_filter_buffer, -1, axis=0)
        
        #update the last element of the array with the new data
        self.Acc_median7_filter_buffer[6] = AccXYZ_curr

        median = np.median(self.Acc_median7_filter_buffer, axis=0)
        return median

    
    def calculate_velocity(self, data_timestamp):
        time_diff_s = (data_timestamp - self.prev_acc_update_timestamp) * self.nanosecs2secs_coef
        
        #use the average accelaration for more accurate integration
        self.Vel_curr = self.Vel_prev + (self.Acc_curr + self.Acc_prev) * 0.5 * time_diff_s
        
        #fill the values for the next cycle
        self.Vel_prev = self.Vel_curr  
        self.prev_vel_update_timestamp = data_timestamp
        

    def calculate_position(self, data_timestamp):
        time_diff_s = (data_timestamp - self.prev_pos_update_timestamp) * self.nanosecs2secs_coef
        
        #use the average velocity for more accurate integration
        self.Pos_curr = self.Pos_prev + (self.Vel_curr + self.Vel_prev) * 0.5 * time_diff_s
        
        #fill the values for the next cycle
        self.Pos_prev = self.Pos_curr 
        self.prev_pos_update_timestamp = data_timestamp

    def __on_acc_error(self, error, e):
        print("error occurred")
        print(e)

    def __on_acc_close(self, ws, a, b):
        print("connection close")

    def __on_acc_open(self, ws):
        print("connection open")

    def calibrate_acc_sensor(self, values, data_timestamp):
        self.Acc_cal_buffer[self.acc_cal_data_cnt] = [values[0], values[1], values[2], data_timestamp]
        self.acc_cal_data_cnt += 1

        if(self.acc_cal_data_cnt >= 5000): #1000 samples 2sn
            
            #sliding median filter (7th order) is applied on raw data to get rid of the spikes. 
            medX = signal.medfilt(self.Acc_cal_buffer[:, 0], kernel_size=7)
            medY = signal.medfilt(self.Acc_cal_buffer[:, 1], kernel_size=7)
            medZ = signal.medfilt(self.Acc_cal_buffer[:, 2], kernel_size=7)

            #sensor offset and stddev calculation on median filtered data 
            self.Acc_sensor_bias[0] = [np.mean(medX, dtype = np.float64), np.mean(medY, dtype = np.float64), np.mean(medZ, dtype = np.float64)]
            self.Acc_sensor_stddev[0] = [np.std(medX, dtype = np.float64), np.std(medY, dtype = np.float64), np.std(medZ, dtype = np.float64)]
            
            #np.savetxt('Acc_cal_buffer.csv', self.Acc_cal_buffer, delimiter=',')
            print("Calibration completed: ", self.Acc_sensor_bias, self.Acc_sensor_stddev)
            #plt.plot(self.Acc_cal_buffer[:,3], self.Acc_cal_buffer[:,0])
            #plt.plot(self.Acc_cal_buffer[:,3])
            #plt.show()
            
            self.acc_median7_filter_init()

            self.prev_acc_update_timestamp = self.Acc_cal_buffer[-1, 3] # init the previous reading timestamp from the last calibration data reading time
            self.prev_vel_update_timestamp = self.prev_acc_update_timestamp
            self.prev_pos_update_timestamp = self.prev_acc_update_timestamp
            self.acc_cal_data_cnt = 0
            self.is_acc_calibrated = True
    
    def process_acc_data(self, values, data_timestamp):
        Acc_read_raw = [values[0], values[1], values[2]]        
        #self.debug_raw_data = np.roll(self.debug_raw_data, -1, axis=0)
        #self.debug_raw_data[499] = Acc_read_raw

        #rectify the raw acc reading with substracting the bias
        Acc_bias_corrected = Acc_read_raw - self.Acc_sensor_bias
        #self.debug_curr_data = np.roll(self.debug_curr_data, -1, axis=0)
        #self.debug_curr_data[499] = Acc_bias_corrected

        Acc_median_filtered = self.acc_median7_filter(Acc_bias_corrected)
        #self.debug_median_filtered_data = np.roll(self.debug_median_filtered_data, -1, axis=0)
        #self.debug_median_filtered_data[499] = Acc_median_filtered

        self.Acc_curr[0] = Acc_median_filtered

        #calculate the position and velocity by numerical integration
        self.calculate_velocity(data_timestamp)
        self.calculate_position(data_timestamp)
        
        self.Acc_prev = self.Acc_curr
        self.prev_acc_update_timestamp = data_timestamp

        #print the calculated values in 1 hz
        self.print_data_cnt += 1
        if(self.print_data_cnt >= 500):
            #print("Calculating...")
            self.print_data_cnt = 0
  
    def __on_acc_message(self, ws, message):
            values = json.loads(message)['values']
            data_timestamp = json.loads(message)['timestamp']
            
            if(self.is_acc_calibrated == False):
                self.calibrate_acc_sensor(values, data_timestamp)
            else:
                self.process_acc_data(values, data_timestamp)

def draw_acc_calculations(screen, mobile_sensor_rec):
    white=(255, 255, 255)
    yellow=(255, 255, 0)
    green=(0, 255, 255)
    orange=(255, 100, 0)

    pg.draw.line(screen, yellow, (250, 30), (250 + -1 * int(mobile_sensor_rec.Acc_curr[0, 0] * 100) , 30), 20) #Ax
    pg.draw.line(screen, green, (250, 60), (250 + int(mobile_sensor_rec.Acc_curr[0, 1] * 100), 60), 20) #Ay
    pg.draw.line(screen, orange, (250, 90), (250 + int(mobile_sensor_rec.Acc_curr[0, 2] * 100), 90), 20) #Az

    #draw velocities
    pg.draw.line(screen, white, (250, 150), (250 + int(mobile_sensor_rec.Vel_curr[0, 0] * 100) , 150), 20) #Vx
    pg.draw.line(screen, orange, (250, 180), (250 + int(mobile_sensor_rec.Vel_curr[0, 1] * 100), 180), 20) #Vy
    pg.draw.line(screen, yellow, (250, 210), (250 + int(mobile_sensor_rec.Vel_curr[0, 2] * 100), 210), 20) #Vz

    #draw positions
    pg.draw.line(screen, white, (250, 270), (250 + -1 * int(mobile_sensor_rec.Pos_curr[0, 0] * 100) , 270), 20) #Vx
    pg.draw.line(screen, green, (250, 300), (250 + int(mobile_sensor_rec.Pos_curr[0, 1] * 100), 300), 20) #Vy
    pg.draw.line(screen, orange, (250, 330), (250 + int(mobile_sensor_rec.Pos_curr[0, 2] * 100), 330), 20) #Vz

def process_user_input(events):
    for event in events:
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                return False
                    
        elif event.type == pg.QUIT:
            return False
    
    return True

if __name__ == "__main__":  
    pg.init()
    screen = pg.display.set_mode([500, 500])
    pg.display.set_caption("Indoor positioning App")

    clock = pg.time.Clock()
    
    mobile_sensor_rec = MobileSensorReceiver()
    mobile_sensor_rec.start()

    while True:
        clock.tick(60)
        
        screen.fill((0,0,0))
        draw_acc_calculations(screen, mobile_sensor_rec)
        
        # Flip the display
        pg.display.flip()

        #check quit request
        if (process_user_input(pg.event.get()) == False):
            break

    pg.quit()
