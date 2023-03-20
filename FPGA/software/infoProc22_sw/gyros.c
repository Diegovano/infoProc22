#include "system.h" // for TIMER_BASE and LED_TIMER_BASE
#include "altera_avalon_timer_regs.h" // for IOWR_*
// #include "altera_up_avalon_accelerometer_spi_regs.h" // for IOWR()
#include "sys/alt_stdio.h" // for alt_putstr()
#include "alt_types.h" // alt_* types
#include "sys/alt_irq.h" // for alt_irq_register()
#include <stdlib.h> // for abs()
#include "altera_up_avalon_accelerometer_spi.h" // for alt_up_accelerometer_spi_open_dev()
#include "stdio.h" // for printf()
#include "stdbool.h" // for true & false
#include "altera_avalon_spi.h" // for alt_avalon_spi_command()
#include "7segformat.h" //for 7 seg
#include "GY271.h" //for magnetometer 
#include "math.h"
#include "system.h" //for floating point
#include "dodgey_trig.h" //decent trig 

#define OFFSET -32
#define PWM_PERIOD 16
#define EST 13
#define TAPS 49
#define TAPS_MAG 20
#define MSEC_SUM_TIMEOUT 5
#define MSEC_UART_TX_TIMEOUT 5
#define MSEC_ESP_TX_TIMEOUT 200
#define STRIDE_LENGTH 0.73

#define INTEGRATE_ON_BOARD false
#define DEBUG_UART true
#define UART false

alt_8 pwm = 0;
alt_u8 led;
alt_u32 timer = 0;
int level;
int pulse; 
int Lightshift;
float heading_roll;

alt_u8 recv[3] = {0};
alt_u8 send[2] = {0};

int stepcount = 0;
alt_u32 last_step_at = 0;

// globals to try to get rid of
double xaccel;
double xpos;
float qfiltered;
alt_u16 acc_sq;

typedef struct position{
  float x;
  float y;
} pos;

typedef struct suvat {
  int acc;
  float vel;
  float pos;
} suvat;

typedef struct dimension {
  int x;
  int y;
  int z;
}dim;

dim mag_bias;
suvat x, y, z;
pos location;

// ====================================
//
// LED
//
// ====================================

void convert_read(alt_32 acc_read, int * level, alt_u8 * led) {
  acc_read += OFFSET;
  alt_u8 val = (acc_read >> 6) & 0x07;
  *led = (8 >> val) | (8 << (8 - val));
  *level = (acc_read >> 1) & 0x1f;
}

void led_write(alt_u8 led_pattern) {
  IOWR(LED_BASE, 0, led_pattern | pulse << 9);
}

// ====================================
//
// FIR
//
// ====================================

float fir_quantised(alt_32 *samples, alt_32 new_sample, unsigned int taps, int *coefficients,int count) {
    samples[count % taps] = new_sample;
    int sum = 0;
    for (unsigned int i = 0; i < taps; i++) {
        sum += coefficients[i] * (int)samples[(count + taps - i) % taps];
    }
    return (float)(sum >> EST);
}

// ====================================
//
// SUMMATION
//
// ====================================

// void summation(int *sum, int summand) {
//   *sum += summand * MSEC_SUM_TIMEOUT;
// }

void summation(float *sum, float summand) {
  *sum += summand * MSEC_SUM_TIMEOUT;
}

// ====================================
//
// BIAS
//
// ====================================

alt_u32 bias(alt_32 *x_samples, alt_32 *y_samples,
alt_32 *z_samples, int *quant_coefficients, alt_up_accelerometer_spi_dev *acc_dev){
  int count = 0;
  float x_bias = 0, y_bias = 0, z_bias = 0;
  alt_32 x_read, y_read, z_read;
  for(int j = 0;j <= 1000; j++){
    count++;
    alt_up_accelerometer_spi_read_x_axis(acc_dev, &x_read);
    alt_up_accelerometer_spi_read_y_axis(acc_dev, &y_read);
    alt_up_accelerometer_spi_read_z_axis(acc_dev, &z_read);
    x_bias += fir_quantised(x_samples, x_read, TAPS, quant_coefficients,count);
    y_bias += fir_quantised(y_samples, y_read, TAPS, quant_coefficients,count);
    z_bias += fir_quantised(z_samples, z_read, TAPS, quant_coefficients,count);
    if(j == TAPS*4){
      x_bias = 0;
      y_bias = 0;
      z_bias = 0;
      count = 0;
    }
  }
  x_bias /= (float)count;
  y_bias /= (float)count;
  z_bias /= (float)count;
  return abs(x_bias * x_bias + y_bias * y_bias + z_bias * z_bias);
}

// ====================================
//
// MAGNETOMETER
//
// ====================================

//read the magnetometer values
void magnetometer_read(alt_16 *x, alt_16 *y, alt_16 *z, alt_u8 *ready){
  *x = GY_271_Read_x();
  *y = GY_271_Read_y();
  *z = GY_271_Read_z();
}

void bias_mag(){
  dim mag_max;
  dim mag_min;
  mag_max.x=-0xffff;mag_max.y=-0xffff;mag_max.z=-0xffff;mag_min.x=0xffff;mag_min.y=0xffff;mag_min.z=0xffff;
  alt_16 x_read_mag, y_read_mag, z_read_mag;
  alt_u8 ready;
  hex_write_left(" ROTATE SLOWLY =]");
  for(int i = 0;i<10000;i++){
    magnetometer_read(&x_read_mag, &y_read_mag, &z_read_mag,&ready);
    if(x_read_mag > mag_max.x) mag_max.x = x_read_mag;
    if(x_read_mag < mag_min.x) mag_min.x = x_read_mag;
    if(y_read_mag > mag_max.y) mag_max.y = y_read_mag;
    if(y_read_mag < mag_min.y) mag_min.y = y_read_mag;
    if(z_read_mag > mag_max.z) mag_max.z = z_read_mag;
    if(z_read_mag < mag_min.z) mag_min.z = z_read_mag;
  }
  mag_bias.x = (mag_max.x + mag_min.x)/2; 
  mag_bias.y = (mag_max.y + mag_min.y)/2; 
  mag_bias.z = (mag_max.z + mag_min.z)/2;
}

void compass_heading(float *samples_x,float* samples_y){
  float sum_x = 0 ,sum_y = 0;
  for(int i = 0;i < TAPS_MAG;i++){
    sum_x += samples_x[i];
    sum_y += samples_y[i];
  }
  heading_roll = atan2(sum_y,sum_x);
}

void compass_direction(float *samples_x,float* samples_y,int count){
  alt_16 x_read_mag, y_read_mag, z_read_mag;
  alt_u8 ready;
  //magnetometer_read(&x_read_mag, &y_read_mag, &z_read_mag,&ready);
  GY_271_Roll_Over_read(&x_read_mag,&y_read_mag,&z_read_mag,&ready);
  x_read_mag = (x_read_mag-mag_bias.x);
  y_read_mag = (y_read_mag-mag_bias.y);
  z_read_mag = (z_read_mag-mag_bias.z);
  float roll = atan2(z.acc, sqrt(x.acc*x.acc + y.acc*y.acc));
  float pitch = atan2(x.acc, sqrt(z.acc*z.acc + y.acc*y.acc));
  float cosRoll = cos(roll);
  float sinRoll = sin(roll);
  float cosPitch = cos(pitch);
  float sinPitch = sin(pitch);
  float xh = x_read_mag * cosPitch + z_read_mag * sinPitch;
  float yh = x_read_mag * sinRoll * sinPitch + y_read_mag * cosRoll - z_read_mag * sinRoll * cosPitch;
  // float sample = atan2(yh, xh);
  // char buffer[5]; 
  // itoa((int)( sample*57.3 +180), buffer, 10);
  // hex_write_left("   ");
  // hex_write_left(buffer);
  samples_y[count % TAPS_MAG] = yh; //heading roll
  samples_x[count % TAPS_MAG] = xh;
  //float heading = 57.3 * atan2((double)x_read_mag,(double)y_read_mag);
  //printf("x:%d y:%d z:%d p:%d r:%d heading:%d\n",x_read_mag,y_read_mag,z_read_mag, (int)(57.3 * roll),(int)(57.3 * pitch),(int) heading_roll + 180);  //just for testing
  //printf("A: %d x, %d y, %d z\n", (int)(x.acc), (int)(y.acc), (int)(z.acc)); 
  //hex_write_left(itoa((int)heading_roll + 180,str,10));
}

// ====================================
//
// INTERRUPT CALLBACKS
//
// ====================================
void timeout_isr() {
  int switches = IORD_16DIRECT(SWITCH_BASE,0); //switches
  IOWR_ALTERA_AVALON_TIMER_STATUS(TIMER_BASE, 0); // reset interrupt
  timer++;
  Lightshift++;
  if (Lightshift % 1000 == 0) shift7seg(); 
  // pulse LED 10
  if (timer % 1000 < 100) pulse = 1; 
  else pulse = 0;
 
  #if INTEGRATE_ON_BOARD
  if (timer % MSEC_SUM_TIMEOUT == 0) {
    summation(&x.vel, abs(x.acc) > 5 ? x.acc / 1000.0f : 0); summation(&x.pos, x.vel / 1000);
    summation(&y.vel, abs(y.acc) > 5 ? y.acc / 1000.0f : 0); summation(&y.pos, y.vel / 1000);
    summation(&z.vel, abs(z.acc) > 5 ? z.acc / 1000.0f : 0); summation(&z.pos, z.vel / 1000);
  }

  if (timer % MSEC_SUM_TIMEOUT * 1 == 0) {
    x.vel = /*x.vel < 0.00000001 ? 0 :*/ x.vel * 0.99;
    y.vel = /*y.vel < 0.00000001 ? 0 :*/ y.vel * 0.99;
    z.vel = /*z.vel < 0.00000001 ? 0 :*/ z.vel * 0.99;
  }
  #else

  if (timer % MSEC_ESP_TX_TIMEOUT == 0) {

    send[0] = stepcount;
    send[1] = heading_roll;

    int RECV_BUFFER_SIZE = 1;
    int SEND_BUFFER_SIZE = 2;

    int length = alt_avalon_spi_command(SPI_BASE, 0, SEND_BUFFER_SIZE, send, SEND_BUFFER_SIZE + RECV_BUFFER_SIZE, recv, 0);

    printf("%d | ", recv[0]);
    printf("\n");

    // if(recv[0] != 0){
    //   char buffer[5]; 
    //   itoa(recv[0], buffer, 10);
    //   hex_write_left(buffer);
    // }
  }

  if (timer % MSEC_UART_TX_TIMEOUT == 0) {
    if ( switches & 1 && timer - last_step_at > 500 && acc_sq > 1000) {
      last_step_at = timer;
      char buffer[5]; 
      location.x += STRIDE_LENGTH*sin(heading_roll);
      location.y += STRIDE_LENGTH*cos(heading_roll);
      itoa(++stepcount, buffer, 10);
      hex_write_left("   ");
      hex_write_right(buffer);
      hex_write_left(buffer);
    }
    // order must be MSB, LSB
    alt_8 header = 0b11000010; // first two bits header, third unsigned, rest represent number of segments per reconstructed type. Here we are reconstructing one variable, with 14 bits transmitted for each.
    // alt_8 payload[] = {(alt_8)((x.acc & 0x3F80) >> 7), (alt_8)(x.acc & 0x7F)/*, (alt_8)(y.acc & 0x7F), (alt_8)((y.acc & 0x3F80) >> 7), (alt_8)(z.acc & 0x7F), (alt_8)((z.acc & 0x3F80) >> 7)*/};
    alt_8 payload[] = {(alt_u8)((acc_sq & 0x3F80) >> 7), (alt_u8)(acc_sq & 0x7F)};
    alt_8 trailer = 0xFF; // trailer, indicate end of stream

    #if UART
    #if DEBUG_UART
    printf("start: %d\n", header);
    #else
    putchar(header);
    #endif
    for (unsigned i = 0; i < sizeof(payload) / sizeof(*payload); i++)
    {
      #if DEBUG_UART
      printf("%d\n", payload[i]);
      #else
      putchar(payload[i]);
      #endif
    }
     #if DEBUG_UART
    printf("end: %d\n", trailer);
    #else
    putchar(trailer);
    #endif
  #endif
  }
  #endif
}

void sys_timer_isr() {
  IOWR_ALTERA_AVALON_TIMER_STATUS(LED_TIMER_BASE, 0); // reset interrupt

  if (pwm < abs(level)) {

    if (level < 0) {
        led_write(led << 1);
    } else { 
        led_write(led >> 1);
    }

  } else {
      led_write(led);
  }

  if (pwm > PWM_PERIOD) {
      pwm = 0;
  } else {
      pwm++;
  }
}

// ====================================
//
// INTERRUPT SETUP
//
// ====================================


void led_timer_init(void (*isr)(void*, long unsigned int)) {
  IOWR_ALTERA_AVALON_TIMER_CONTROL(LED_TIMER_BASE, 0x0003);
  IOWR_ALTERA_AVALON_TIMER_STATUS(LED_TIMER_BASE, 0);
  IOWR_ALTERA_AVALON_TIMER_PERIODL(LED_TIMER_BASE, 0x0900);
  IOWR_ALTERA_AVALON_TIMER_PERIODH(LED_TIMER_BASE, 0x0000);
  alt_irq_register(LED_TIMER_IRQ, 0, isr);
  IOWR_ALTERA_AVALON_TIMER_CONTROL(LED_TIMER_BASE, 0x0007);
}

void timer_init(void (*isr)(void*, long unsigned int)) {
  IOWR_ALTERA_AVALON_TIMER_CONTROL(TIMER_BASE, 0x0003);
  IOWR_ALTERA_AVALON_TIMER_STATUS(TIMER_BASE, 0);
  IOWR_ALTERA_AVALON_TIMER_PERIODL(TIMER_BASE, 0xC350); // corresponds to 1ms because Bourganis said 1s is roughly 0x2FAF080
  IOWR_ALTERA_AVALON_TIMER_PERIODH(TIMER_BASE, 0x0000);
  alt_irq_register(TIMER_IRQ, 0, isr);
  IOWR_ALTERA_AVALON_TIMER_CONTROL(TIMER_BASE, 0x0007);
}

// ====================================
//
// MAIN
//
// ====================================


alt_u16 accel_abs_sq(suvat x, suvat y, suvat z, alt_u32 grav_bias) {
  return abs(x.acc * x.acc + y.acc * y.acc + z.acc * z.acc - grav_bias) >> 5;
}


int main()
{
  hex_write_left("STILL!");
  alt_32 x_read_acc, y_read_acc, z_read_acc;
  float samples_mag_x[TAPS_MAG];
  float samples_mag_y[TAPS_MAG];
  alt_u16 switches;
  alt_u8 buttons;
  alt_32 *x_samples = calloc(TAPS, sizeof(alt_32));
  alt_32 *y_samples = calloc(TAPS, sizeof(alt_32));
  alt_32 *z_samples = calloc(TAPS, sizeof(alt_32)); 
  alt_up_accelerometer_spi_dev *acc_dev;
  acc_dev = alt_up_accelerometer_spi_open_dev("/dev/accelerometer_spi"); 

  if (acc_dev == NULL) { // if return 1, check if the spi ip name is "accelerometer_spi"
      return 1;
  }

  float coefficients[] = {0.0046, 0.0074, -0.0024, -0.0071, 0.0033, 0.0001, -0.0094, 0.0040, 0.0044, -0.0133,
                          0.0030, 0.0114, -0.0179, -0.0011, 0.0223,-0.0225, -0.0109, 0.0396,-0.0263, -0.0338,
                          0.0752,-0.0289, -0.1204,  0.2879, 0.6369, 0.2879, -0.1204,-0.0289, 0.0752, -0.0338,
                         -0.0263, 0.0396, -0.0109, -0.0225, 0.0223,-0.0011, -0.0179, 0.0114, 0.0030, -0.0133,
                          0.0044, 0.0040, -0.0094,  0.0001, 0.0033,-0.0071, -0.0024, 0.0074, 0.0046};

  int coef_size = sizeof(coefficients) / sizeof(coefficients[0]);

  int *quant_coefficients = calloc(coef_size, sizeof(int));

  for (int i = 0; i < coef_size; i++) {
    quant_coefficients[i] = (int)(coefficients[i] * (1<<EST)); // closest power of 2, could be faster than multiplying by 10000
  }

  // REGISTER INTERRUPTS
  led_timer_init(sys_timer_isr);
  timer_init(timeout_isr);

  

  int count = 0, count2 =0;
  char str[32];
  GY_271_init(MAGNETOMETER_BASE,50000000,100000);
  GY_271_Reset(); 
  GY_271_setMode(1,0,2,1,1,1); 
  //bias(&x_bias, &y_bias, &z_bias, x_samples, y_samples, z_samples, quant_coefficients, acc_dev);
  alt_u32 grav_bias = bias(x_samples, y_samples, z_samples, quant_coefficients, acc_dev);
  bias_mag();
  hex_write_clear();
  // hex_write_left("AAAAA");
  while (1) {
    buttons = (~IORD_8DIRECT(BUTTON_BASE,0))&0b11; //buttons 
    x.acc = (int)(fir_quantised(x_samples, x_read_acc, TAPS, quant_coefficients,count)) /*& 0xFFFFFFF8*/; // remove LS 2 bits effect
    y.acc = (int)(fir_quantised(y_samples, y_read_acc, TAPS, quant_coefficients,count)) /*& 0xFFFFFFF8*/;
    z.acc = (int)(fir_quantised(z_samples, z_read_acc, TAPS, quant_coefficients,count)) /*& 0xFFFFFFF8*/;
    alt_up_accelerometer_spi_read_x_axis(acc_dev, &x_read_acc);
    alt_up_accelerometer_spi_read_y_axis(acc_dev, &y_read_acc);
    alt_up_accelerometer_spi_read_z_axis(acc_dev, &z_read_acc);

    acc_sq = accel_abs_sq(x, y, z, grav_bias);

    if (count % 10 == 0) {
      compass_direction(samples_mag_x,samples_mag_y,count2);
      count2++;
    }

    if (count % 100 == 0) {
      compass_heading(samples_mag_x,samples_mag_y);
      char buffer[5]; 
      itoa((int)(heading_roll*57.3 +180), buffer, 10);
      hex_write_right("   ");
      hex_write_right(buffer);
    }

    if(buttons&0b1){
      bias_mag();
      hex_write_clear();
    }

    if(buttons&0b10){
      grav_bias = bias(x_samples, y_samples, z_samples, quant_coefficients, acc_dev);
      hex_write_clear();
    }
    // convert_read(x.acc, &level, &led);

    // alt_u8 msg = (alt_u8)x.acc;

    count++;

    if (count % 100 == 0) {
      // printf("A: %d x, %d y, %d z\tV: %d x, %d y, %d z \tP: %d x, %d y, %d z\n", (int)(x.acc), (int)(y.acc), (int)(z.acc), (int)(x.vel), (int)(y.vel), (int)(z.vel), (int)(x.pos), (int)(y.pos), (int)(z.pos));
    }
  }
  return 0;
}
