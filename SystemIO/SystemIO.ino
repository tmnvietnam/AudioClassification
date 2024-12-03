

#include "IOPin.h"

uint8_t responseBytes[] = {0x44, 0x45, 0x53, 0x00, 0x52, 0x56};

void setup()
{
  Serial.begin(9600);
  SetSystemIOPinMode();
}

void loop()
{
  CollectInput();
}

// runs whenever there's serial buffer
// System control event
void serialEvent()
{
  uint8_t bytesData[20];
  if (Serial.available())
  {
    delay(10);
    uint8_t readdedbyte = Serial.read();
    if (readdedbyte == 0x44)
    {
      int index = 0;
      bytesData[index] = readdedbyte;
      while (Serial.available())
      {
        readdedbyte = Serial.read();
        index++;
        bytesData[index] = readdedbyte;
        if (index >= 9)
        {
          // check the command
          if (bytesData[3] == 0x4F)
          {
            // get the data bytes
            uint8_t bytes[4] = {bytesData[4], bytesData[5], bytesData[6], bytesData[7]};
            SetSystemOutput(bytes);
            Serial.write(systemResponseOutput, 6);
          }
          break;
        }
       
      }
    }
    while (Serial.available())
    {
      readdedbyte = Serial.read();
    }
  }
}