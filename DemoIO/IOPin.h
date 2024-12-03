#define STARTIN 2
#define RESETIN 3
#define RESETOUT 22

void SetSystemIOPinMode()
{
    Serial.begin(9600);
    pinMode(STARTIN, INPUT);
    pinMode(RESETIN, INPUT);
    pinMode(RESETOUT, OUTPUT);
}


