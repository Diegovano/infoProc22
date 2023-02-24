#include "system.h" // for TIMER_BASE and LED_TIMER_BASE
#include "altera_avalon_timer_regs.h" // for IOWR_*
// #include "altera_up_avalon_accelerometer_spi_regs.h" // for IOWR()
#include "sys/alt_stdio.h" // for alt_putstr()
#include "alt_types.h" // alt_* types
#include "sys/alt_irq.h" // for alt_irq_register()
#include <stdlib.h> // for abs()
#include "altera_up_avalon_accelerometer_spi.h" // for alt_up_accelerometer_spi_open_dev()
#include "stdio.h" // for printf()

#define OFFSET -32
#define PWM_PERIOD 16

alt_8 pwm = 0;
alt_u8 led;
alt_u32 timer = 0;
int level;
int pulse;


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

float fir_quantised(alt_32 *samples, alt_32 new_sample, unsigned int taps, int *coefficients) {
    for (unsigned int i = taps - 1; i > 0; i--) {
        samples[i] = samples[i - 1];
    }
    samples[0] = new_sample;

    int sum = 0;
    for (unsigned int i = 0; i < taps; i++) {
        sum += coefficients[i] * (int)samples[i];

    }
    // return sum / 8192.0f;
    return (float)(sum >> 13);
    // return sum / 10000.0f;
}

// ====================================
//
// INTERRUPTS
//
// ====================================

// callbacks

void timeout_isr() {
  IOWR_ALTERA_AVALON_TIMER_STATUS(TIMER_BASE, 0); // reset interrupt
  timer++;

  if (timer % 1000 < 100) pulse = 1;
  else pulse = 0;
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

// setup

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
  const int TAPS = 49;
  alt_32 x_read;
  alt_32 *samples = calloc(TAPS, sizeof(alt_32));
  alt_up_accelerometer_spi_dev *acc_dev;
  acc_dev = alt_up_accelerometer_spi_open_dev("/dev/accelerometer_spi");

  if (acc_dev == NULL) { // if return 1, check if the spi ip name is "accelerometer_spi"
      return 1;
  }

  float qfiltered;
  float coefficients[] = {0.0046, 0.0074, -0.0024, -0.0071, 0.0033, 0.0001, -0.0094, 0.0040, 0.0044, -0.0133,
                          0.0030, 0.0114, -0.0179, -0.0011, 0.0223,-0.0225, -0.0109, 0.0396,-0.0263, -0.0338,
                          0.0752,-0.0289, -0.1204,  0.2879, 0.6369, 0.2879, -0.1204,-0.0289, 0.0752, -0.0338,
                         -0.0263, 0.0396, -0.0109, -0.0225, 0.0223,-0.0011, -0.0179, 0.0114, 0.0030, -0.0133,
                          0.0044, 0.0040, -0.0094,  0.0001, 0.0033,-0.0071, -0.0024, 0.0074, 0.0046};

  int coef_size = sizeof(coefficients) / sizeof(coefficients[0]);

  int *quant_coefficients = calloc(coef_size, sizeof(int));

  for (int i = 0; i < coef_size; i++) {
    quant_coefficients[i] = (int)(coefficients[i] * 8192); // closest power of 2, could be faster than multiplying by 10000
  }

  // REGISTER INTERRUPTS
  led_timer_init(sys_timer_isr);
  timer_init(timeout_isr);

  int count = 0;

  while (1) {
    qfiltered = fir_quantised(samples, x_read, 49, quant_coefficients);

    alt_up_accelerometer_spi_read_x_axis(acc_dev, &x_read);

    convert_read(qfiltered, &level, &led);

    count++;

    if (count % 100 == 0) {
      printf("A: %d\n"/*,\tV: %d,\tP: %d\n"*/, (int)(qfiltered)/*, (int)(xaccel), (int)(xpos)*/);
    }
  }
  return 0;
}
