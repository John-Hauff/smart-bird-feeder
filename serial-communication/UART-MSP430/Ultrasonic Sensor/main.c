#include "msp430.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define TRIG_PIN BIT1 // Corresponds to P2.1
#define ECHO_PIN BIT0 // Corresponds to P2.0

#define TXD BIT2 // TXD on P1.2

volatile unsigned long start_time;
volatile unsigned long end_time;
volatile unsigned long delta_time;
volatile unsigned long distance;
volatile unsigned long sum;
char ch;
volatile int i;

void print(char *text)
{
  unsigned int i = 0;
  while (text[i] != '\0')
  {
    while (!(IFG2 & UCA0TXIFG))
      ; // Check if TX is ongoing
    UCA0TXBUF = text[i];
    i++;
  }
}

void printNumber(unsigned long num)
{
  char buf[8];
  char *str = &buf[7];

  *str = '\0';

  do
  {
    unsigned long m = num;
    num /= 10;
    char c = (m - 10 * num) + '0';
    *--str = c;
  } while (num);

  print(str);
}

void wait_ms(unsigned int ms)
{
  unsigned int i;
  for (i = 0; i <= ms; i++)
  {
    __delay_cycles(1000); // 1MHz clock --> 1E3/1E6 = 1E-3 (1ms)
  }
}

#if defined(__TI_COMPILER_VERSION__)
#pragma vector = TIMER1_A0_VECTOR
__interrupt void ta1_isr(void)
#else
void __attribute__((interrupt(TIMER1_A0_VECTOR))) ta1_isr(void)
#endif
{
  switch (TA1IV) // Interrupt Vector Register (determines which flag requested interrupt)
  {
  case 10: // 0Ah = 10 = Timer Overflow
    break;

  default:              // Capture Interrupt (High if receiving input)
    if (TA1CCTL0 & CCI) // CCI bit - capture/compare input in CCTL0
    {
      start_time = TA1CCR0; // start time of receiving pulse
    }
    else
    {
      end_time = TA1CCR0;
      delta_time = end_time - start_time;
      distance = (unsigned long)(delta_time / 0.00583090379); // micrometers
      if (distance / 10000 >= 2.0 && distance / 10000 <= 400) // HC-SR04 acceptible measure ranges
      {
        sum += distance;
      }
    }
  }
  TACTL &= ~CCIFG; // reset the interrupt flag
}

/* Setup TRIGGER and ECHO pins */
void init_ultrasonic_pins(void)
{
  P2DIR &= ~ECHO_PIN; // Set ECHO (P2.0) pin as INPUT
  P2SEL |= ECHO_PIN;  // Set P2.0 as CCI0A (Capture Input signal).
  P2SEL2 &= ~(ECHO_PIN);
  P2DIR |= TRIG_PIN;  // Set TRIGGER (P2.1) pin as OUTPUT
  P2OUT &= ~TRIG_PIN; // Set TRIGGER (P2.1) pin to LOW
}

// Initialize the motors
void init_motor(void)
{
  // The P1.6 is used for pwm and outpu to the servo.
  P1DIR |= BIT6;
  // This clears all P1 outputs.
  P1OUT = 0;
  // Selecting the TA0.1 option.
  P1SEL |= BIT6;
}

void delay()
{
  volatile unsigned long i;
  i = 49999;
  do
    (i--);
  while (i != 0);
}

/* Setup UART */
void init_uart(void)
{
  P1SEL = BIT1 + BIT2; // Select UART RX/TX function on P1.1,P1.2
  P1SEL2 = BIT1 + BIT2;

  UCA0CTL1 |= UCSSEL_2; // Use SMCLK - 1MHz clock
  UCA0BR0 = 104;        // Set baud rate to 9600 with 1MHz clock
  UCA0BR1 = 0;          // Set baud rate to 9600 with 1MHz clock
  UCA0MCTL = UCBRS0;    // Modulation UCBRSx = 1
  UCA0CTL1 &= ~UCSWRST; // Initialize USCI state machine - enable
  IE2 |= UCA0RXIE;      // Enable RX interrupt
}

void init_timer(void)
{
  // 1MHz Clock
  BCSCTL1 = CALBC1_1MHZ; // Set range
  DCOCTL = CALDCO_1MHZ;
  BCSCTL2 &= ~(DIVS_3); // SMCLK = DCO = 1MHz

  TA1CTL = MC_0; // Stop timer before modifying configuration

  // CM_3 = Capture on both rising and falling edges
  // SCS = Synchronize capture source - might make asynchronous?
  // CCIS_0 = CCI0A
  // CAP = Capture Mode
  // CCIE = Interrupt enabled
  TA1CCTL0 |= CM_3 + SCS + CCIS_0 + CAP + CCIE;

  // TASSEL_2 = SMCLK
  // MC_2 = Continuous Mode
  // ID_0 = /1 divider
  TA1CTL |= TASSEL_2 + MC_2 + ID_0;
}

void reset_timer(void)
{
  TA1CTL |= TACLR; // Clear timer
}

#pragma vector = USCIAB0RX_VECTOR // UART RX Interrupt Vector
__interrupt void USCI0RX_ISR(void)
{
  char c;
  c = UCA0RXBUF;
  if (c == 'u')
  {
    sum = 0;
    i = 0;
    while (i < 6)
    {
      __enable_interrupt(); // Global Interrupt Enable
      reset_timer();
      P2OUT |= TRIG_PIN;  // Start of Pulse
      __delay_cycles(10); // Send pulse for 10us
      P2OUT &= ~TRIG_PIN; // End of Pulse
      wait_ms(1000);      // = 1 seconds
      i = i + 1;
    }
    sum = sum / 6;
    if (sum < 60000)
    {
      print("h");
    }
    if (sum > 59999)
    {
      print("l");
    }

    IFG2 &= ~(UCA0RXIFG);               // Clear Receiver flag
    __bis_SR_register(LPM0_bits + GIE); // Enter LPM0, Enable Interrupt
  }

  if (c == 'o')
  {
    // Set PWM period TA0.1.
    TA0CCR0 = 20000 - 1;
    // This sets 1.5 ms as 0 degrees, servo position.
    TA0CCR1 = 1500;

    TA0CCTL1 = OUTMOD_7;
    TA0CTL = TASSEL_2 + MC_1;

    // This block moves the motor 90 degrees to open position
    delay();
    TA0CCR1 = 1500;
    delay();
    // TA0CCR1 = 1750;
    // delay();
    TA0CCR1 = 2000;
    delay();
    // TA0CCR1 = 2250;
    // delay();
    TA0CCR1 = 2500;
    delay();

    TA0CTL = MC_0;

    IFG2 &= ~(UCA0RXIFG);               // Clear Receiver flag
    __bis_SR_register(LPM0_bits + GIE); // Enter LPM0, Enable Interrupt
  }

  if (c == 'c')
  {
    // Set PWM period TA0.1.
    TA0CCR0 = 20000 - 1;
    // This sets 2.5 ms as 90 degrees, servo position.
    TA0CCR1 = 2500;

    TA0CCTL1 = OUTMOD_7;
    TA0CTL = TASSEL_2 + MC_1;

    // This block rotates the servo 90 deg to close hatch
    delay();
    TA0CCR1 = 2500;
    delay();
    // TA0CCR1 = 2250;
    // delay();
    TA0CCR1 = 2000;
    delay();
    // TA0CCR1 = 1750;
    // delay();
    TA0CCR1 = 1500;
    delay();

    TA0CTL = MC_0;

    IFG2 &= ~(UCA0RXIFG);               // Clear Receiver flag
    __bis_SR_register(LPM0_bits + GIE); // Enter LPM0, Enable Interrupt
  }
}

void main(void)
{
  WDTCTL = WDTPW + WDTHOLD; // Stop Watch Dog Timer

  if (CALBC1_1MHZ == 0xFF) // Check if calibration constant erased
  {
    while (1)
      ; // do not load program
  }
  DCOCTL = 0;            // Select lowest DCO settings
  BCSCTL1 = CALBC1_1MHZ; // Set DCO to 1 MHz
  DCOCTL = CALDCO_1MHZ;

  init_ultrasonic_pins();
  init_uart();
  init_timer();
  init_motor();

  __bis_SR_register(LPM0_bits + GIE); // Enter LPM0, Enable Interrupt
}