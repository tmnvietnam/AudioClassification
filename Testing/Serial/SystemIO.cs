using System.Collections;
using System.Windows.Shapes;

namespace SoundKit.Serial
{
    public class SystemIO : Device
    {


        public SystemIO(Rectangle connectionStatusDevice, Rectangle txStatusDevice, Rectangle rxStatusDevice) : base(connectionStatusDevice, txStatusDevice, rxStatusDevice)
        {

        }

        public event EventHandler OnStartRequest;


        // state define
        public const bool ON = true;
        public const bool OFF = false;

        // Machine SYSTEM OUTPUT
        /// <summary>
        /// Software sw control main cylinder going down
        /// </summary>
        private bool _Lock;

        public bool Lock
        {
            get { return _Lock; }
            set
            {
                if (_Lock != value)
                {
                    _Lock = value;
                }
 
            }
        }


        /// <summary>
        /// Software switch lock card on top
        /// </summary>
        private bool _Card_TOP_LOCK;
        public bool Card_TOP_LOCK
        {
            get { return _Card_TOP_LOCK; }
            set
            {
                if (value != _Card_TOP_LOCK) _Card_TOP_LOCK = value;
            }
        }

        /// <summary>
        /// Software switch lock card on bot
        /// </summary>
        private bool _Card_BOT_LOCK;
        public bool Card_BOT_LOCK
        {
            get { return _Card_BOT_LOCK; }
            set
            {
                if (value != _Card_BOT_LOCK) _Card_BOT_LOCK = value;
            }
        }

        /// <summary>
        /// Software switch lock JIG on top
        /// </summary>
        private bool _JIG_TOP_LOCK;
        public bool JIG_TOP_LOCK
        {
            get { return _JIG_TOP_LOCK; }
            set
            {
                if (value != _JIG_TOP_LOCK) _JIG_TOP_LOCK = value;
            }
        }

        /// <summary>
        /// Software switch lock JIG on top
        /// </summary>
        private bool _JIG_BOT_LOCK;
        public bool JIG_BOT_LOCK
        {
            get { return _JIG_BOT_LOCK; }
            set
            {
                if (value != _JIG_BOT_LOCK) _JIG_BOT_LOCK = value;
            }
        }

        /// <summary>
        /// Tower lamps RED light
        /// </summary>
        private bool _LPR;
        public bool LPR
        {
            get { return _LPR; }
            set
            {
                if (value != _LPR)
                    _LPR = value;
                if (value)
                {
                    LPG = false;
                    LPY = false;
                }
            }
        }

        /// <summary>
        /// Tower lamps YELLOW light
        /// </summary>
        private bool _LPY;
        public bool LPY
        {
            get { return _LPY; }
            set
            {
                if (value != _LPY)
                    _LPY = value;
                if (value)
                {
                    LPG = false;
                    LPR = false;
                }
            }
        }

        /// <summary>
        /// Tower lamps GREEN light
        /// </summary>
        private bool _LPG;
        public bool LPG
        {
            get { return _LPG; }
            set
            {
                if (value != _LPG)
                    _LPG = value;
                if (value)
                {
                    LPR = false;
                    LPY = false;
                    BUZZER = false;
                }
            }
        }

        /// <summary>
        /// Tower lamps Buzzer
        /// </summary>
        private bool _BUZZER;
        public bool BUZZER
        {
            get { return _BUZZER; }
            set
            {
                if (value != _BUZZER) _BUZZER = value;
            }
        }

        /// <summary>
        /// AC power 110V site A/C, on power and on AC0....load to site (A,C)
        /// </summary>
        private bool _AC110;
        public bool AC110
        {
            get { return _AC110; }
            set
            {
                if (value != _AC110) _AC110 = value;
                if (value)
                {
                    ADSC = false;
                    AC220 = false;
                    AC0 = true;
                }
            }
        }




        /// <summary>
        /// AC power on site A/C, on power and on AC0....load to site (A,C)
        /// </summary>
        private bool _AC0;

        public bool AC0
        {
            get { return _AC0; }
            set
            {
                if (value != _AC0) _AC0 = value;
                if (value)
                    ADSC = false;
            }
        }

        /// <summary>
        /// AC power 220V site A/C, on power and on AC0....load to site (A,C)
        /// </summary>
        private bool _AC220;

        public bool AC220
        {
            get { return _AC220; }
            set
            {
                if (value != _AC220)
                    _AC220 = value;
                if (value)
                {
                    ADSC = false;
                    AC110 = false;
                    AC0 = true;
                }
            }
        }

        /// <summary>
        /// Discharge site A/C
        /// </summary>
        private bool _ADSC1;

        public bool ADSC1
        {
            get { return _ADSC1; }
            set
            {
                if (value != _ADSC1)
                    _ADSC1 = value;
            }
        }

        /// <summary>
        /// Discharge site A/C
        /// </summary>
        private bool _ADSC2;

        public bool ADSC2
        {
            get { return _ADSC2; }
            set
            {
                if (value != _ADSC2) _ADSC2 = value;
            }
        }

        public bool ADSC
        {
            get { return ADSC1 || ADSC2; }
            set
            {
                ADSC1 = value;
                ADSC2 = value;
                if (value)
                {
                    AC0 = false;
                    AC110 = false;
                    AC220 = false;
                }
            }
        }

        /// <summary>
        /// AC power 110V site B/D, on power and on BC0....load to site (B,D)
        /// </summary>
        private bool _BC110;

        public bool BC110
        {
            get { return _BC110; }
            set
            {
                if (value != _BC110) _BC110 = value;
                if (value)
                {
                    BC220 = false;
                    BDSC = false;
                    BC0 = true;
                }
            }
        }

        /// <summary>
        /// AC power on site B/D, on power and on BC0....load to site (B,D)
        /// </summary>
        private bool _BC0;

        public bool BC0
        {
            get { return _BC0; }
            set
            {
                if (value != _BC0) _BC0 = value;
                if (value)
                    BDSC = false;
            }
        }

        /// <summary>
        /// AC power 220V site B/D, on power and on BC0....load to site (B,D)
        /// </summary>
        private bool _BC220;

        public bool BC220
        {
            get { return _BC220; }
            set
            {
                if (value != _BC220) _BC220 = value;
                if (value)
                {
                    BDSC = false;
                    BC110 = false;
                    BC0 = true;
                }
            }
        }

        /// <summary>
        /// Discharge site B/D
        /// </summary>
        private bool _BDSC1;

        public bool BDSC1
        {
            get { return _BDSC1; }
            set
            {
                if (value != _BDSC1) _BDSC1 = value;
            }
        }

        /// <summary>
        /// Discharge site B/D
        /// </summary>
        private bool _BDSC2;

        public bool BDSC2
        {
            get { return _BDSC2; }
            set
            {
                if (value != _BDSC2) _BDSC2 = value;
            }
        }

        public bool BDSC
        {
            get { return BDSC1 || BDSC2; }
            set
            {
                BDSC1 = value;
                BDSC2 = value;
                if (value)
                {
                    BC0 = false;
                    BC110 = false;
                    BC220 = false;
                }
                
            }
        }

        // Machine SYSTEM GEN
        /// <summary>
        /// Generation a frequency
        /// </summary>
        private Int32 _A1_GEN;

        public Int32 A1_GEN
        {
            get { return _A1_GEN; }
            set
            {
                if (value != _A1_GEN) _A1_GEN = value;
            
            }
        }

        /// <summary>
        /// Generation a frequency
        /// </summary>
        private Int32 _A2_GEN;

        public Int32 A2_GEN
        {
            get { return _A2_GEN; }
            set
            {
                if (value != _A2_GEN) _A2_GEN = value;
            }
        }

        /// <summary>
        /// Generation a frequency
        /// </summary>
        private Int32 _B1_GEN;

        public Int32 B1_GEN
        {
            get { return _B1_GEN; }
            set
            {
                if (value != _B1_GEN) _B1_GEN = value;
            }
        }

        /// <summary>
        /// Generation a frequency
        /// </summary>
        private Int32 _B2_GEN;

        public Int32 B2_GEN
        {
            get { return _B2_GEN; }
            set
            {
                if (value != _B2_GEN) _B2_GEN = value;
            }
        }

        // Machine SYSTEM INPUT
        /// <summary>
        /// Switch main cylinder Up on machine
        /// </summary>
        private bool _SW_UP;

        public bool SW_UP
        {
            get { return _SW_UP; }
            set
            {
                if (value != _SW_UP)
                    _SW_UP = value;
                Lock = value;
            }
        }

        /// <summary>
        /// Switch Main cylinder down on machine
        /// </summary>
        private bool _SW_DOWN;

        public bool SW_DOWN
        {
            get { return _SW_DOWN; }
            set
            {
                if (value != _SW_DOWN)
                {
                    _SW_DOWN = value;
                }
                //MainDOWN = value;
            }
        }

        /// <summary>
        /// switch top card release
        /// </summary>
        private bool _SW_BR;

        public bool SW_BR
        {
            get { return _SW_BR; }
            set
            {
                if (value != _SW_BR) _SW_BR = value;
            }
        }

        /// <summary>
        /// Switch bot card insert
        /// </summary>
        private bool _SW_BF;

        public bool SW_BF
        {
            get { return _SW_BF; }
            set
            {
                if (value != _SW_BF)
                    _SW_BF = value;
            }
        }

        /// <summary>
        /// Switch top card release
        /// </summary>
        private bool _SW_TR;

        public bool SW_TR
        {
            get { return _SW_TR; }
            set
            {
                if (value != _SW_TR)
                    _SW_TR = value;
            }
        }

        /// <summary>
        /// Switch top cart Insert
        /// </summary>
        private bool _SW_TF;

        public bool SW_TF
        {
            get { return _SW_TF; }
            set
            {
                if (value != _SW_TF)
                    _SW_TF = value;
            }
        }

        /// <summary>
        /// Emc button
        /// </summary>
        private bool _SW_EMC;

        public bool SW_EMC
        {
            get { return _SW_EMC; }
            set
            {
                if (value != _SW_EMC)
                    _SW_EMC = value;

            }
        }

        public bool NotEMC
        {
            get { return !_SW_EMC; }
        }


        /// <summary>
        /// Sensor main cylinder upstate
        /// </summary>
        private bool _SS_UP;

        public bool SS_UP
        {
            get { return _SS_UP; }
            set
            {
                if (value != _SS_UP) _SS_UP = value;
            }
        }

        /// <summary>
        /// Sensor main cylinder down state
        /// </summary>
        private bool _SS_DOWN;

        public bool SS_DOWN
        {
            get { return _SS_DOWN; }
            set
            {
                if (value != _SS_DOWN)
                {
                    if (_SS_DOWN == OFF)
                    {
                        OnStartRequest?.Invoke(null, null);
                    }
                    if (_SS_DOWN == ON)
                    {
                        //OnCancleRequest?.Invoke(null, null);
                    }
                    //_SS_DOWN = value;
                }
                //_MainDOWN = value;
       
                //OnUpDown?.Invoke("DOWN", null);
            }
        }

        /// <summary>
        /// Sensor card release on Bot side
        /// </summary>
        private bool _SS_BR;

        public bool SS_BR
        {
            get { return _SS_BR; }
            set
            {
                if (value != _SS_BR) _SS_BR = value;
            }
        }

        /// <summary>
        /// Sensor card inserted on Bot side
        /// </summary>
        private bool _SS_BF;

        public bool SS_BF
        {
            get { return _SS_BF; }
            set
            {
                if (value != _SS_BF) _SS_BF = value;
                //Card_BOT_LOCK = value;
            }
        }

        /// <summary>
        /// Sensor card release on Top side
        /// </summary>
        private bool _SS_TR;

        public bool SS_TR
        {
            get { return _SS_TR; }
            set
            {
                if (value != _SS_TR) _SS_TR = value;
            }
        }

        /// <summary>
        /// Sensor card inserted on Top side
        /// </summary>
        private bool _SS_TF;

        public bool SS_TF
        {
            get { return _SS_TF; }
            set
            {
                if (value != _SS_TF) _SS_TF = value;
                //Card_TOP_LOCK = value;
            }
        }

        /// <summary>
        /// Sensor lock JIG on Bot side
        /// </summary>
        private bool _SS_BL;

        public bool SS_BL
        {
            get { return _SS_BL; }
            set
            {
                if (value != _SS_BL)
                    _SS_BL = value;
                //JIG_BOT_LOCK = value;
            }
        }

        /// <summary>
        /// Sensor JIG locked on Top side
        /// </summary>
        private bool _SS_TL;

        public bool SS_TL
        {
            get { return _SS_TL; }
            set
            {
                if (value != _SS_TL)
                    _SS_TL = value;
                //JIG_TOP_LOCK = value;
            }
        }

        public void SendControl()
        {
            var data = DataInput();
            if (SerialPort.IsOpen)
            {
                SendBytes(CreateFrame(data));
            }
        }

        public void DataOuput(byte[] bytes)
        {
            if (bytes.Length != 4)
            {
                return;
            }
            uint Data32Bit = BitConverter.ToUInt32(bytes, 0);

            //InputIO
            SW_UP = GetValue(Data32Bit, 0);
            SW_DOWN = GetValue(Data32Bit, 1);
            SW_BR = GetValue(Data32Bit, 2);
            SW_BF = GetValue(Data32Bit, 3);
            SW_TR = GetValue(Data32Bit, 4);
            SW_TF = GetValue(Data32Bit, 5);
            SW_EMC = GetValue(Data32Bit, 6);
            SS_UP = GetValue(Data32Bit, 8);
            SS_DOWN = GetValue(Data32Bit, 9);
            SS_BR = GetValue(Data32Bit, 10);
            SS_BF = GetValue(Data32Bit, 11);
            SS_TR = GetValue(Data32Bit, 12);
            SS_TF = GetValue(Data32Bit, 13);
            SS_BL = GetValue(Data32Bit, 14);
            SS_TL = GetValue(Data32Bit, 15);      
        }

        public bool GetValue(uint data, int position)
        {
            return (data & (uint)(1 << position)) != 0;
        }

        public byte[] DataInput()
        {
            byte[] bytes = new byte[5];
            bytes[0] = 0x4F;
            List<bool> OutPuts = new List<bool>
            {
                AC110,
                AC0,
                AC220,
                ADSC1,
                ADSC2,
                false,
                false,
                false,

                BC110,
                BC0,
                BC220,
                BDSC1,
                BDSC2,
                false,
                false,
                false,

                LPR,
                LPY,
                LPG,
                BUZZER,
                false,
                false,
                false,
                false,

                Lock,
                false,
                false,
                false,
                false,
                false,
                false,
                false
            };

            BitArray bits = new BitArray(OutPuts.ToArray());
            bits.CopyTo(bytes, 1);
            return bytes;
        }


    }

}
