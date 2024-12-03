#include "IOPin.h"

void setup()
{
  Serial.begin(9600);
  SetSystemIOPinMode();
}

void loop()
{
  uint8_t startButton = digitalRead(STARTIN);
  uint8_t resetButton = digitalRead(RESETIN);

  Serial.print("Start Button: ");
  Serial.print(startButton);
  Serial.print(", Reset Button: ");
  Serial.println(resetButton);

  if (Serial.available()) {          // Check if data is received
    char command = Serial.read();    // Read the incoming byte
    if (command == '1') {            // If '1' is received
      digitalWrite(RESETOUT, HIGH);    // Turn the LED on
    } else if (command == '0') {     // If '0' is received
      digitalWrite(RESETOUT, LOW);     // Turn the LED off
    }
  }
}