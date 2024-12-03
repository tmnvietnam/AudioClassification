#define RESET 3      
#define START 2    
#define ResetOUT 22 

void SetSystemIOPinMode() {
  Serial.begin(9600);

  pinMode(START, INPUT);
  pinMode(RESET, INPUT);
  pinMode(ResetOUT, OUTPUT);
}

uint8_t systemResponseInput[10] = { 0x44, 0x45, 0x06, 0x49, 0x00, 0xE8, 0x00, 0x00, 0x52, 0x56 };
uint8_t systemResponseOutput[10] = { 0x44, 0x45, 0x4F, 0x00, 0x52, 0x56 };
uint8_t LastStartState = false;
uint8_t LastResetState = false;

void CollectInput() {
  uint8_t startSignal = digitalRead(START);
  uint8_t resetSignal = digitalRead(RESET);

  

  if (startSignal != LastStartState) {
    delay(500);
    startSignal = digitalRead(START);
    if (startSignal != LastStartState) {
     
      LastStartState = startSignal;
      bitWrite(systemResponseInput[5], 1, startSignal);

      unsigned char xorTemp = systemResponseInput[0];
      for (int i = 1; i < 8; i++) {
        xorTemp ^= systemResponseInput[i];
      }
      systemResponseInput[8] = xorTemp;

      Serial.write(systemResponseInput, 10);
    }
  }

  if (resetSignal != LastResetState) {
    delay(500);
    resetSignal = digitalRead(RESET);
    if (resetSignal != LastResetState) {
      LastResetState = resetSignal;
      bitWrite(systemResponseInput[5], 0, resetSignal);

      unsigned char xorTemp = systemResponseInput[0];
      for (int i = 1; i < 8; i++) {
        xorTemp ^= systemResponseInput[i];
      }
      systemResponseInput[8] = xorTemp;

      Serial.write(systemResponseInput, 10);
    }
  }
}


void SetSystemOutput(uint8_t data[4]) {
  uint32_t data32 = 0x00000000;

  for (int index = 0; index < 4; index++) {
    data32 = data32 << 8 | data[index];
  }

  digitalWrite(ResetOUT, bitRead(data32, 0));
}
