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
        self.Acc_sensor_bias = np.zeros((1, 3), dtype = np.float64)
        self.Acc_sensor_stddev = np.zeros((1, 3), dtype = np.float64)
        self.Acc_cal_buffer = np.zeros((2000, 4), dtype = np.float64)


        self.ws = websocket.WebSocketApp("ws://192.168.41.17:8081/sensor/connect?type=android.sensor.accelerometer",
                                            on_open=self.__on_acc_open,
                                            on_message=self.__on_acc_message,
                                            on_error=self.__on_acc_error,
                                            on_close=self.__on_acc_close)

        self.acc_cal_data_cnt = 0
        self.is_acc_calibrated = False
        self.nanosecs2secs_coef = 1. / 1.
        self.print_data_cnt = 0
        self.gui_text = ""

        #new algoritm variables
        self.AccX = 0.
        self.VelX = 0.
        self.PosX = 0.
        self.acc_cum_x = 0.
        self.acc_data_cnt = 0
        self.acc_prev_x = 0.
        self.vel_prev_x = 0.
        self.pos_prev_x = 0.
        self.acc_zero_cnt = 0

 
    def process_acc_data(self, acc_read_x, acc_read_y, acc_read_z):
        # Sampling freq is 500Hz periodic
        # function developed considered the performance
        if (self.acc_data_cnt < 10):
            self.acc_cum_x = self.acc_cum_x + acc_read_x;
            self.acc_data_cnt += 1
            return

        
        #10Hz loop
        acc_x = self.acc_cum_x / 10.0  - self.Acc_sensor_bias[0, 0]
        self.acc_data_cnt = 0
        self.acc_cum_x = 0.0

        if abs(acc_x) < 0.03:   #ignore the noise
            acc_x = 0.0
            self.acc_zero_cnt +=1
        else:
            self.acc_zero_cnt = 0


        #first integration for the velocity applying trapezoid:
        #dt = 2 * self.nanosecs2secs_coef, 0.5 * dt = self.nanosecs2secs_coef

        dt = 2 * self.nanosecs2secs_coef
        vel_x = self.vel_prev_x + (self.acc_prev_x + ((acc_x - self.acc_prev_x) * 0.5)) * dt;
        
        #second X integration applying trapezoid:
        pos_x = self.pos_prev_x + (self.vel_prev_x + ((vel_x - self.vel_prev_x) * 0.5)) * dt;

        #update the prev values for the next loop
        self.acc_prev_x = acc_x  
        self.vel_prev_x = vel_x
        self.pos_prev_x = pos_x


        #if acc value is 0 for sometime, movement is finished
        # set the velocity to zero
        if (self.acc_zero_cnt >= 10):  #1s
            vel_x = 0
            self.vel_prev_x = 0
            self.acc_zero_cnt = 0
      

        self.AccX = acc_x
        self.VelX = vel_x
        self.PosX = pos_x

        #update the gui msg @ 1 hz
        self.print_data_cnt += 1
        if(self.print_data_cnt >= 10):
            self.gui_text = "Processing.."
            self.print_data_cnt = 0
  
    def __on_acc_message(self, ws, message):
            values = json.loads(message)['values']
            data_timestamp = json.loads(message)['timestamp']
            
            if(self.is_acc_calibrated == False):
                self.calibrate_acc_sensor(values, data_timestamp)
            else:
                self.process_acc_data(values[0], values[1], values[2])

    def calibrate_acc_sensor(self, values, data_timestamp):
        self.Acc_cal_buffer[self.acc_cal_data_cnt] = [values[0], values[1], values[2], data_timestamp]
        self.acc_cal_data_cnt += 1
        self.gui_text = "Calibrating"

        if(self.acc_cal_data_cnt == 2000): #1000 samples 2sn
            
            accX = self.Acc_cal_buffer[:, 0]
            accY = self.Acc_cal_buffer[:, 1]
            accZ = self.Acc_cal_buffer[:, 2]

            #sensor offset and stddev calculation on median filtered data 
            self.Acc_sensor_bias[0] = [np.mean(accX, dtype = np.float64), np.mean(accY, dtype = np.float64), np.mean(accZ, dtype = np.float64)]
            self.Acc_sensor_stddev[0] = [np.std(accX, dtype = np.float64), np.std(accY, dtype = np.float64), np.std(accZ, dtype = np.float64)]
            
            #np.savetxt('Acc_cal_buffer.csv', self.Acc_cal_buffer, delimiter=',')
            self.gui_text = "Calibration completed"
            print("Calibration values: ", self.Acc_sensor_bias, self.Acc_sensor_stddev)
   
            self.acc_cal_data_cnt = 0
            self.is_acc_calibrated = True
    
    def start(self):
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def __on_acc_error(self, error, e):
        self.gui_text = "Error occured" + str(e)
        print(self.gui_text)

    def __on_acc_close(self, ws, a, b):
        self.gui_text = "Connection closed"
        print(self.gui_text)

    def __on_acc_open(self, ws):
        self.gui_text = "Connection opened"
        print(self.gui_text)

def draw_acc_debug_text(screen, mobile_sensor_rec):
    font = pg.font.SysFont(None, 24)

    # now print the text
    text_surface = font.render(mobile_sensor_rec.gui_text, True, (255, 255, 255))
    screen.blit(text_surface,(20,470))

def draw_acc_calculations(screen, mobile_sensor_rec):
    white=(255, 255, 255)
    yellow=(255, 255, 0)
    green=(0, 255, 255)
    orange=(255, 100, 0)

    pg.draw.line(screen, yellow, (250, 30), (250 + -1 * int(mobile_sensor_rec.AccX * 100) , 30), 20) #Ax
    pg.draw.line(screen, green, (250, 60), (250 + -1 * int(mobile_sensor_rec.AccX) * 100, 60), 20) #Ay
    pg.draw.line(screen, orange, (250, 90), (250 + -1 * int(mobile_sensor_rec.AccX) * 100, 90), 20) #Az

    #draw velocities
    pg.draw.line(screen, white, (250, 150), (250 +  int(mobile_sensor_rec.VelX * 100) , 150), 20) #Vx
    pg.draw.line(screen, orange, (250, 180), (250 + int(mobile_sensor_rec.VelX * 100), 180), 20) #Vy
    pg.draw.line(screen, yellow, (250, 210), (250 + int(mobile_sensor_rec.VelX * 100), 210), 20) #Vz

    #draw positions
    pg.draw.line(screen, white, (250, 270), (250 + int(mobile_sensor_rec.PosX * 10) , 270), 20) #Vx
    pg.draw.line(screen, green, (250, 300), (250 + int(mobile_sensor_rec.PosX * 10), 300), 20) #Vy
    pg.draw.line(screen, orange, (250, 330), (250 + int(mobile_sensor_rec.PosX * 10), 330), 20) #Vz

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
    pg.font.init()
    screen = pg.display.set_mode([500, 500])
    pg.display.set_caption("Indoor positioning App")

    clock = pg.time.Clock()
    
    mobile_sensor_rec = MobileSensorReceiver()
    mobile_sensor_rec.start()

    while True:
        clock.tick(60)
        
        screen.fill((0,0,0))
        draw_acc_calculations(screen, mobile_sensor_rec)
        draw_acc_debug_text(screen, mobile_sensor_rec)
        
        # Flip the display
        pg.display.flip()

        #check quit request
        if (process_user_input(pg.event.get()) == False):
            break

    pg.quit()
