#include "msp430.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define TRIG_PIN BIT1   // Corresponds to P2.1
#define ECHO_PIN BIT0   // Corresponds to P2.0

#define TXD BIT2        // TXD on P1.2

volatile unsigned long start_time;
volatile unsigned long end_time;
volatile unsigned long delta_time;
volatile unsigned long distance;
volatile unsigned long sum;
char ch;
volatile int i;
unsigned int ADC_value=0;

void print(char *text)
{
  unsigned int i = 0;
  while(text[i] != '\0')
  {
    while (!(IFG2 & UCA0TXIFG));        // Check if TX is ongoing
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
    } while(num);

    print(str);
}

void wait_ms(unsigned int ms)
{
  unsigned int i;
  for (i = 0; i <= ms; i++)
  {
    __delay_cycles(1000);             //1MHz clock --> 1E3/1E6 = 1E-3 (1ms)
  }
}

void ConfigureAdc(void)
{
    ADC10CTL1 = INCH_3 + ADC10DIV_3;
    ADC10CTL0 = SREF_0 + ADC10SHT_3 + ADC10ON + ADC10IE;
    ADC10AE0 |= BIT3;
}

#if defined(__TI_COMPILER_VERSION__)
#pragma vector = TIMER1_A0_VECTOR
__interrupt void ta1_isr(void)
#else
void __attribute__((interrupt(TIMER1_A0_VECTOR))) ta1_isr(void)
#endif
{
  switch (TA1IV)                        // Interrupt Vector Register (determines which flag requested interrupt)
  {
  case 10:                              // 0Ah = 10 = Timer Overflow
    break;

  default:                              // Capture Interrupt (High if receiving input)
    if (TA1CCTL0 & CCI)                 // CCI bit - capture/compare input in CCTL0
    {
      start_time = TA1CCR0;             // start time of receiving pulse
    }
    else
    {
      end_time = TA1CCR0;
      delta_time = end_time - start_time;
      distance = (unsigned long)(delta_time / 0.00583090379);       // micrometers
      if (distance / 10000 >= 2.0 && distance / 10000 <= 400)       // HC-SR04 acceptible measure ranges
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
  P2DIR &= ~ECHO_PIN;             // Set ECHO (P1.1) pin as INPUT
  P2SEL |= ECHO_PIN;              // Set P1.1 as CCI0A (Capture Input signal).
  P2SEL2 &= ~(ECHO_PIN);
  P2DIR |= TRIG_PIN;              // Set TRIGGER (P2.1) pin as OUTPUT
  P2OUT &= ~TRIG_PIN;             // Set TRIGGER (P2.1) pin to LOW
}

void init_motor(void){
  P1DIR |= BIT6;
  P1SEL |= BIT6;
  P1OUT = 0;
}

void delay(){
  volatile unsigned long i;
  i = 49999;
  do (i--);
  while (i != 0);
}

/* Setup UART */
void init_uart(void)
{
  P1SEL = BIT1 + BIT2 ;               // Select UART RX/TX function on P1.1,P1.2
  P1SEL2 = BIT1 + BIT2;


  UCA0CTL1 |= UCSSEL_2;               // Use SMCLK - 1MHz clock
  UCA0BR0 = 104;                      // Set baud rate to 9600 with 1MHz clock
  UCA0BR1 = 0;                        // Set baud rate to 9600 with 1MHz clock
  UCA0MCTL = UCBRS0;                  // Modulation UCBRSx = 1
  UCA0CTL1 &= ~UCSWRST;               // Initialize USCI state machine - enable
  IE2 |= UCA0RXIE;                    // Enable RX interrupt
}

void init_adc(void)
{
    P1SEL |= BIT3;                    // ADC Input
}

void init_speaker(void)
{
    P2DIR |= BIT2;
    P2SEL |= BIT2;
    P2OUT |= 0;
}

void init_timer(void)
{
  // 1MHz Clock
  BCSCTL1 = CALBC1_1MHZ;        // Set range
  DCOCTL = CALDCO_1MHZ;
  BCSCTL2 &= ~(DIVS_3);         // SMCLK = DCO = 1MHz

  TA1CTL = MC_0;                // Stop timer before modifying configuration

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
  TA1CTL |= TACLR;                      //Clear timer
}

// ADC10 interrupt service routine
#pragma vector=ADC10_VECTOR
__interrupt void ADC10_ISR (void)
{
    __bic_SR_register_on_exit(CPUOFF);        // Return to active mode; BIC mask, saved_SR
}

#pragma vector=USCIAB0RX_VECTOR         // UART RX Interrupt Vector
__interrupt void USCI0RX_ISR(void)
{
  char c;
  c = UCA0RXBUF;

  if(c == 'u'){
    sum = 0;
    i = 0;
    while( i < 6) {
      __enable_interrupt();         // Global Interrupt Enable
      reset_timer();
      P2OUT |= TRIG_PIN;            // Start of Pulse
      __delay_cycles(10);           // Send pulse for 10us
      P2OUT &= ~TRIG_PIN;           // End of Pulse
      wait_ms(1000);                // = 1 seconds
      i = i + 1;
    }
    sum = sum / 6;
    if (sum < 60000){
        print("h");
    }
    if (sum > 59999){
        print("l");
    }

    IFG2 &= ~(UCA0RXIFG);                  // Clear Receiver flag
   // __bis_SR_register(LPM0_bits + GIE);    // Enter LPM0, Enable Interrupt
  }

  if(c == 'o'){
    TA0CCR0 = 20000-1;
    TA0CCR1 = 1500;

    TA0CCTL1 = OUTMOD_7;
    TA0CTL = TASSEL_2 + MC_1;

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

    IFG2 &= ~(UCA0RXIFG);                  // Clear Receiver flag
    // __bis_SR_register(LPM0_bits + GIE);    // Enter LPM0, Enable Interrupt

  }

  if(c == 'c'){
    //Start of motor code
    TA0CCR0 = 20000-1;
    TA0CCR1 = 2500;

    TA0CCTL1 = OUTMOD_7;
    TA0CTL = TASSEL_2 + MC_1;

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

    init_timer();

    IFG2 &= ~(UCA0RXIFG);                  // Clear Receiver flag
    // __bis_SR_register(LPM0_bits + GIE);    // Enter LPM0, Enable Interrupt
  }
}

void main(void)
{
  WDTCTL = WDTPW + WDTHOLD;             // Stop Watch Dog Timer

  if (CALBC1_1MHZ == 0xFF)                // Check if calibration constant erased
  {
      while(1);                         // do not load program
  }
  DCOCTL = 0;                           // Select lowest DCO settings
  BCSCTL1 = CALBC1_1MHZ;                // Set DCO to 1 MHz
  DCOCTL = CALDCO_1MHZ;

  init_ultrasonic_pins();
  init_uart();
  init_timer();
  init_motor();
  init_speaker();

  //ADC Code
  init_adc();

  // ADC Code
  ConfigureAdc();

  // ADC Code
  BCSCTL2 &= ~(DIVS_3);

  __enable_interrupt();                 // enable global interrupt

  // Loop for trip-wire
  while(1)
  {
    wait_ms(3000);                      // wait every 1s to check voltage
    ADC10CTL0 |= ENC + ADC10SC;         // sampling and conversion; enable and start conversion
    __bis_SR_register(CPUOFF + GIE);    // enter LPM mode; exits when ADC10_ISR is triggered
    ADC_value = ADC10MEM;               // ADC10MEM = ADC10 Memory
   // printNumber(ADC_value);
    if(ADC_value > 200)
    {
        print("r");
    }
  }
  // __bis_SR_register(LPM0_bits + GIE);   // Enter LPM0, Enable Interrupt
}

