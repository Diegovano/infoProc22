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
#include "7segformat.h"
#include "i2c_opencores.h"
#include "i2c_opencores_regs.h"

#define OFFSET -32
#define PWM_PERIOD 16
#define EST 13
#define TAPS 49
#define MSEC_SUM_TIMEOUT 5
#define MSEC_TX_TIMEOUT 5

#define INTEGRATE_ON_BOARD false
#define DEBUG_UART true

alt_8 pwm = 0;
alt_u8 led;
alt_u32 timer = 0;
int level;
int pulse; 
int Lightshift;

// globals to try to get rid of
double xaccel;
double xpos;
float qfiltered;

typedef struct dimension {
  int acc;
  float vel;
  float pos;
} dim;

dim x, y, z;

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

void bias(float *x_bias, float *y_bias, float *z_bias, alt_32 *x_samples, alt_32 *y_samples,
alt_32 *z_samples, int *quant_coefficients, alt_up_accelerometer_spi_dev *acc_dev){
  int count = 0;
  alt_32 x_read, y_read, z_read;
  for(int j = 0;j <= 1000; j++){
    count++;
    alt_up_accelerometer_spi_read_x_axis(acc_dev, &x_read);
    alt_up_accelerometer_spi_read_y_axis(acc_dev, &y_read);
    alt_up_accelerometer_spi_read_z_axis(acc_dev, &z_read);
    *x_bias += fir_quantised(x_samples, x_read, TAPS, quant_coefficients,count);
    *y_bias += fir_quantised(y_samples, y_read, TAPS, quant_coefficients,count);
    *z_bias += fir_quantised(z_samples, z_read, TAPS, quant_coefficients,count);
    if(j == TAPS*4){
      *x_bias = 0;
      *y_bias = 0;
      *z_bias = 0;
      count = 0;
    }
  }
  *x_bias /= (float)count;
  *y_bias /= (float)count;
  *z_bias /= (float)count;
}

// ====================================
//
// MAGNETOMETER
//
// ====================================

void magnetometer(){
  alt_16 x,y,z;
  I2C_start(MAGNETOMETER_BASE,0xd,0);
  I2C_write(MAGNETOMETER_BASE,0x0,1);
  I2C_start(MAGNETOMETER_BASE,0xd,1);
  x = I2C_read(MAGNETOMETER_BASE,0);
  x |= I2C_read(MAGNETOMETER_BASE,0)<<8;
  y = I2C_read(MAGNETOMETER_BASE,0);
  y |= I2C_read(MAGNETOMETER_BASE,0)<<8;
  z = I2C_read(MAGNETOMETER_BASE,0);
  z |= I2C_read(MAGNETOMETER_BASE,1)<<8;
  printf("x:%d y:%d z:%d\n",x,y,z);
}

// ====================================
//
// INTERRUPT CALLBACKS
//
// ====================================
void timeout_isr() {
  IOWR_ALTERA_AVALON_TIMER_STATUS(TIMER_BASE, 0); // reset interrupt
  int mag;
  timer++;
  Lightshift++;
  if (Lightshift % 1000 == 0) shift7seg();
  if (timer > 1000 && timer%100 ==0) magnetometer();
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

  if (timer % MSEC_TX_TIMEOUT == 0) {

    // order must be MSB, LSB
    alt_8 header = 0b11000010; // first two bits header, rest represent number of segments per reconstructed type. Here we are reconstructing three variables, with seven bits transmitted for each.
    alt_8 payload[] = {(alt_8)((x.acc & 0x3F80) >> 7), (alt_8)(x.acc & 0x7F)/*, (alt_8)(y.acc & 0x7F), (alt_8)((y.acc & 0x3F80) >> 7), (alt_8)(z.acc & 0x7F), (alt_8)((z.acc & 0x3F80) >> 7)*/};
    alt_8 trailer = 0xFF; // trailer, indicate end of stream

    #if DEBUG_UART
    //printf("start: %d\n", header);
    #else
    putchar(header);
    #endif
    for (unsigned i = 0; i < sizeof(payload) / sizeof(*payload); i++)
    {
      #if DEBUG_UART
      //printf("%d\n", payload[i]);
      #else
      putchar(payload[i]);
      #endif
    }
     #if DEBUG_UART
    //printf("end: %d\n", trailer);
    #else
    putchar(trailer);
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


int main()
{
  hex_write("test....test....=]....."); 
  alt_32 x_read, y_read, z_read;
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
  switches = IORD_16DIRECT(SWITCH_BASE,0); //switches
  buttons = (~IORD_8DIRECT(BUTTON_BASE,0))&0b11; //buttons 
  I2C_init(MAGNETOMETER_BASE,50000000,100000);
  int count = 0;
  int on = 1;
  float x_bias, y_bias, z_bias;
  I2C_start(MAGNETOMETER_BASE,0xd,0);
  I2C_write(MAGNETOMETER_BASE,0x09,0);
  I2C_write(MAGNETOMETER_BASE,0x1D,1);
  //bias(&x_bias, &y_bias, &z_bias, x_samples, y_samples, z_samples, quant_coefficients, acc_dev);
  while (1) {
    x.acc = (int)(fir_quantised(x_samples, x_read, TAPS, quant_coefficients,count) - x_bias) /*& 0xFFFFFFF8*/; // remove LS 2 bits effect
    y.acc = (int)(fir_quantised(y_samples, y_read, TAPS, quant_coefficients,count) - y_bias) /*& 0xFFFFFFF8*/;
    z.acc = (int)(fir_quantised(z_samples, z_read, TAPS, quant_coefficients,count) - z_bias) /*& 0xFFFFFFF8*/;
    alt_up_accelerometer_spi_read_x_axis(acc_dev, &x_read);
    alt_up_accelerometer_spi_read_y_axis(acc_dev, &y_read);
    alt_up_accelerometer_spi_read_z_axis(acc_dev, &z_read);

    

    convert_read(x.acc, &level, &led);

    alt_u8 msg = (alt_u8)x.acc;

    count++;

    if (count % 100 == 0) {
      // printf("A: %d x, %d y, %d z\tV: %d x, %d y, %d z \tP: %d x, %d y, %d z\n", (int)(x.acc), (int)(y.acc), (int)(z.acc), (int)(x.vel), (int)(y.vel), (int)(z.vel), (int)(x.pos), (int)(y.pos), (int)(z.pos));
    }
  }
  return 0;
}