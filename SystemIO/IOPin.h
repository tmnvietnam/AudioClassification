#define SDOWN 3 // sensor bottom level aka start signal
#define AC220 22 // A site 220V power
#define AC110 23 // A site 110V power
#define LPY 29    // Tower lamp yellow
#define LPG 30    // Tower lamp green
#define LPR 31    // Tower lamp red
#define BZ 32     // Tower lamp buzzer
#define CLUP LED_BUILTIN // Cilynder Up control aka reset

void SetSystemIOPinMode()
{
    Serial.begin(9600);

    pinMode(SDOWN, INPUT_PULLUP);

    pinMode(LPG, OUTPUT);
    pinMode(LPY, OUTPUT);
    pinMode(LPR, OUTPUT);
    pinMode(BZ, OUTPUT);

    pinMode(AC110, OUTPUT);
    pinMode(AC220, OUTPUT);



    pinMode(CLUP, OUTPUT);
}

uint8_t systemResponseInput[10] = {0x44, 0x45, 0x06, 0x49, 0x00, 0xE8, 0x00, 0x00, 0x52, 0x56};
uint8_t systemResponseOutput[10] = {0x44, 0x45, 0x4F, 0x00, 0x52, 0x56};
uint8_t LastStartState = false;

void CollectInput()
{
    uint8_t startSignal = digitalRead(SDOWN);

    if (startSignal != LastStartState)
    {
        delay(500);
        startSignal = digitalRead(SDOWN);
        if (startSignal != LastStartState)
        {
            // if (startSignal == 1)
            // {
            //     digitalWrite(LPR, LOW);
            //     digitalWrite(LPY, HIGH);
            //     digitalWrite(LPG, LOW);
            // }
            // else
            // {
            //     digitalWrite(AC110, LOW);
            //     digitalWrite(AC220, LOW);
 
            // }

            LastStartState = startSignal;
            bitWrite(systemResponseInput[5], 1, startSignal);
            unsigned char xorTemp = systemResponseInput[0];
            for (int i = 1; i < 8; i++)
            {
                xorTemp ^= systemResponseInput[i];
            }
            systemResponseInput[8] = xorTemp;

            Serial.write(systemResponseInput, 10);
        }
    }
}

void ResponseInput()
{
    uint8_t startSignal = digitalRead(SDOWN);

    bitWrite(systemResponseInput[5], 1, startSignal);
    unsigned char xorTemp = systemResponseInput[0];
    for (int i = 1; i < 8; i++)
    {
        xorTemp ^= systemResponseInput[i];
    }
    systemResponseInput[8] = xorTemp;
    Serial.write(systemResponseInput, 10);
}

void SetSystemOutput(uint8_t data[4])
{
    uint32_t data32 = 0x00000000;

    for (int index = 0; index < 4; index++)
    {
        data32 = data32 << 8 | data[index];
    }
    // digitalWrite(LPR, bitRead(data32, 8));
    // digitalWrite(LPY, bitRead(data32, 9));
    // digitalWrite(LPG, bitRead(data32, 10));
    // digitalWrite(BZ, bitRead(data32, 11));

    digitalWrite(CLUP, bitRead(data32, 0));
    // if (digitalRead(SDOWN))
    //     digitalWrite(CLUP, bitRead(data32, 0));
    // else
    //     digitalWrite(CLUP, LOW);

    // if (digitalRead(SDOWN))
    // {
    //     digitalWrite(AC110, bitRead(data32, 24));
    //     digitalWrite(AC220, bitRead(data32, 26));
    // }
}
