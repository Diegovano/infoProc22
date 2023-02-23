#include "system.h" // for TIMER_BASE and LED_TIMER_BASE
#include "altera_avalon_timer_regs.h" // for IOWR_*
#include "altera_up_avalon_accelerometer_spi_regs.h" // for IOWR()
#include "sys/alt_stdio.h" // for alt_putstr()
#include "alt_types.h" // alt_* types
#include "sys/alt_irq.h" // for alt_irq_register()
// #include <stdlib.h> // for abs()
#include <cstdlib>

#define PWM_PERIOD 16

alt_8 pwm = 0;
alt_u8 led;
alt_u32 timer = 0;
int level;
int pulse;

// ====================================
//
// INTERRUPTS
//
// ====================================

// callbacks

void timeout_isr() {
  IOWR_ALTERA_AVALON_TIMER_STATUS(TIMER_BASE, 0); // reset interrupt
  timer++;
  
  if(timer % 1000 == 0) alt_putstr("Hello from Nios II!\n");
}

void led_write(alt_u8 led_pattern) {
  IOWR(LED_BASE, 0, led_pattern | pulse << 9);
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
}

int main()
{ 
  while (1);

  return 0;
}
