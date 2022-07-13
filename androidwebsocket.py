from threading import Thread
from time import sleep
import websocket
import json
import numpy as np
from matplotlib import pyplot as plt

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

        self.ws = websocket.WebSocketApp("ws://192.168.41.17:8081/sensor/connect?type=android.sensor.accelerometer",
                                            on_open=self.__on_acc_open,
                                            on_message=self.__on_acc_message,
                                            on_error=self.__on_acc_error,
                                            on_close=self.__on_acc_close)

        self.prev_read_timestamp = 0
        self.acc_cal_data_cnt = 0
        self.is_acc_calibrated = False
        self.nanosecs2secs_coef = 1. / 1000000000

        self.print_data_cnt = 0

    
    def calculate_velocity(self, Acc_curr, time_diff_s):
        return self.Vel_prev + Acc_curr * time_diff_s

    def calculate_position(self, Vel_curr, time_diff_s):
        return self.Pos_prev + Vel_curr * time_diff_s

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
            self.Acc_sensor_bias[0] = [np.mean(self.Acc_cal_buffer[:, 0], dtype = np.float64), np.mean(self.Acc_cal_buffer[:, 1], dtype = np.float64), np.mean(self.Acc_cal_buffer[:, 2], dtype = np.float64)]
            self.Acc_sensor_stddev[0] = [np.std(self.Acc_cal_buffer[:, 0], dtype = np.float64), np.std(self.Acc_cal_buffer[:, 1], dtype = np.float64), np.std(self.Acc_cal_buffer[:, 2], dtype = np.float64)]
            
            np.savetxt('Acc_cal_buffer.csv', self.Acc_cal_buffer, delimiter=',')
            print("Calibration completed: ", self.Acc_sensor_bias, self.Acc_sensor_stddev)
            #plt.plot(self.Acc_cal_buffer[:,3], self.Acc_cal_buffer[:,0])
            #plt.plot(self.Acc_cal_buffer[:,3])
            #plt.show()
            
            self.prev_read_timestamp = self.Acc_cal_buffer[-1, 3] # init the previous reading timestamp from the last calibration data reading time

            self.acc_cal_data_cnt = 0
            self.is_acc_calibrated = True
    
    def process_acc_data(self, values, data_timestamp):
        Acc_read_raw = [values[0], values[1], values[2]]
        
        #correct the raw acc reading with substracting the bias
        Acc_curr = Acc_read_raw - self.Acc_sensor_bias
        
        change = abs(Acc_curr[0, 0] - self.Acc_prev[0, 0])
        
        #ignore the small oscilations on acc to filter out noise
        if (change > (2 * self.Acc_sensor_stddev[0, 0])):
            #calculate the position and velocity by numerical integration
            time_diff_s = (data_timestamp - self.prev_read_timestamp) * self.nanosecs2secs_coef
            self.Vel_curr = self.calculate_velocity(Acc_curr, time_diff_s)
            self.Pos_curr = self.calculate_position(self.Vel_curr, time_diff_s)
            self.Vel_prev = self.Vel_curr  #fill the values for the next cycle
            self.Pos_prev = self.Pos_curr

        else:
            #print("0")
            pass

        #in any case update the time tamp
        self.prev_read_timestamp = data_timestamp

        #print the calculated values in 1 hz
        self.print_data_cnt += 1
        if(self.print_data_cnt >= 100):
            print("Vx(cm/s): Px(cm)", int(self.Vel_curr[0,0] * 100), int(self.Pos_curr[0,0] * 100))
            self.print_data_cnt = 0
  
    def __on_acc_message(self, ws, message):
            values = json.loads(message)['values']
            data_timestamp = json.loads(message)['timestamp']
            
            if(self.is_acc_calibrated == False):
                self.calibrate_acc_sensor(values, data_timestamp)
            else:
                self.process_acc_data(values, data_timestamp)

if __name__ == "__main__":
    mobile_sensor_rec = MobileSensorReceiver()
    mobile_sensor_rec.ws.run_forever()